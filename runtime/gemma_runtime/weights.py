from __future__ import annotations

import argparse
import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass(frozen=True)
class TensorRef:
    name: str
    shard: str
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    data_offsets: tuple[int, int] | None = None
    data_start: int | None = None
    path: Path | None = None

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total if self.shape else 0

    @property
    def nbytes(self) -> int | None:
        if self.data_offsets:
            return self.data_offsets[1] - self.data_offsets[0]
        if self.dtype in DTYPE_BYTES:
            return self.numel * DTYPE_BYTES[self.dtype]
        return None

    @property
    def absolute_offsets(self) -> tuple[int, int] | None:
        if self.data_start is None or self.data_offsets is None:
            return None
        return self.data_start + self.data_offsets[0], self.data_start + self.data_offsets[1]


class SafetensorsShard:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._header: dict[str, Any] | None = None
        self._data_start: int | None = None

    def header(self) -> dict[str, Any]:
        if self._header is not None:
            return self._header
        with self.path.open("rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                raise ValueError(f"{self.path} is too small to be a safetensors file")
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len <= 0 or header_len > 256 * 1024 * 1024:
                raise ValueError(f"{self.path} has invalid safetensors header length {header_len}")
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                raise ValueError(f"{self.path} has truncated safetensors header")
        header = json.loads(header_bytes.decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError(f"{self.path} safetensors header is not an object")
        self._header = header
        self._data_start = 8 + int(header_len)
        return header

    @property
    def data_start(self) -> int:
        if self._data_start is None:
            self.header()
        assert self._data_start is not None
        return self._data_start

    def tensor_ref(self, name: str) -> TensorRef:
        info = self.header().get(name)
        if not isinstance(info, dict):
            raise KeyError(f"{name} not found in {self.path}")
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{name} in {self.path} has invalid data_offsets")
        shape = info.get("shape", [])
        if not isinstance(shape, list):
            raise ValueError(f"{name} in {self.path} has invalid shape")
        return TensorRef(
            name=name,
            shard=self.path.name,
            dtype=info.get("dtype") if isinstance(info.get("dtype"), str) else None,
            shape=tuple(int(v) for v in shape),
            data_offsets=(int(offsets[0]), int(offsets[1])),
            data_start=self.data_start,
            path=self.path,
        )

    def read_tensor_bytes(self, name: str) -> bytes:
        ref = self.tensor_ref(name)
        absolute = ref.absolute_offsets
        if absolute is None:
            raise ValueError(f"{name} has no absolute offsets")
        start, end = absolute
        with self.path.open("rb") as f:
            f.seek(start)
            return f.read(end - start)

    def mmap_tensor_view(self, name: str):
        ref = self.tensor_ref(name)
        absolute = ref.absolute_offsets
        if absolute is None:
            raise ValueError(f"{name} has no absolute offsets")
        f = self.path.open("rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        start, end = absolute
        return f, mm, memoryview(mm)[start:end], ref


class GemmaWeightIndex:
    def __init__(
        self,
        index_path: str | Path,
        shard_roots: Iterable[str | Path] = (),
    ):
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Safetensors index not found: {self.index_path}. "
                "Download the Gemma safetensors files, pass --index explicitly, "
                "or use scripts/gemma_substrate_inspector.py for metadata-only planning."
            )
        with self.index_path.open("r", encoding="utf-8") as f:
            self.index = json.load(f)
        weight_map = self.index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{self.index_path} missing object field weight_map")
        self.weight_map: dict[str, str] = {
            str(name): str(shard)
            for name, shard in weight_map.items()
        }
        roots = [Path(root) for root in shard_roots]
        roots.append(self.index_path.parent)
        self.shard_roots = roots
        self._shard_paths: dict[str, Path] = {}
        self._shards: dict[str, SafetensorsShard] = {}

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "GemmaWeightIndex":
        model_dir = Path(model_dir)
        return cls(model_dir / "model.safetensors.index.json", [model_dir])

    @property
    def tensor_names(self) -> list[str]:
        return sorted(self.weight_map.keys())

    @property
    def shard_names(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    def resolve_shard_path(self, shard_name: str) -> Path:
        if shard_name in self._shard_paths:
            return self._shard_paths[shard_name]
        for root in self.shard_roots:
            candidate = root / shard_name
            if candidate.exists():
                self._shard_paths[shard_name] = candidate
                return candidate
            matches = list(root.rglob(shard_name)) if root.exists() else []
            if matches:
                self._shard_paths[shard_name] = matches[0]
                return matches[0]
        raise FileNotFoundError(f"Shard not found: {shard_name}; searched {', '.join(str(r) for r in self.shard_roots)}")

    def shard(self, shard_name: str) -> SafetensorsShard:
        if shard_name not in self._shards:
            self._shards[shard_name] = SafetensorsShard(self.resolve_shard_path(shard_name))
        return self._shards[shard_name]

    def tensor_ref(self, name: str, require_shard: bool = True) -> TensorRef:
        shard_name = self.weight_map.get(name)
        if shard_name is None:
            raise KeyError(f"Tensor not found in index: {name}")
        if not require_shard:
            return TensorRef(name=name, shard=shard_name)
        return self.shard(shard_name).tensor_ref(name)

    def read_tensor_bytes(self, name: str) -> bytes:
        shard_name = self.weight_map.get(name)
        if shard_name is None:
            raise KeyError(f"Tensor not found in index: {name}")
        return self.shard(shard_name).read_tensor_bytes(name)

    def tensors_for_layer(self, layer: int) -> list[str]:
        language_needles = (
            f".language_model.layers.{layer}.",
            f".language_model.model.layers.{layer}.",
        )
        language = sorted(name for name in self.weight_map if any(needle in name for needle in language_needles))
        if language:
            return language
        fallback_needles = (
            f".layers.{layer}.",
            f".blocks.{layer}.",
            f".h.{layer}.",
        )
        return sorted(name for name in self.weight_map if any(needle in name for needle in fallback_needles))

    def tensor_groups_for_layer(self, layer: int) -> dict[str, list[str]]:
        groups = {"attention": [], "mlp": [], "norm": [], "router": [], "other": []}
        for name in self.tensors_for_layer(layer):
            lower = name.lower()
            if any(part in lower for part in ("self_attn", ".attn", "attention", ".q_proj", ".k_proj", ".v_proj", ".o_proj")):
                groups["attention"].append(name)
            elif any(part in lower for part in ("mlp", "feed_forward", "ffn", ".gate_proj", ".up_proj", ".down_proj")):
                groups["mlp"].append(name)
            elif any(part in lower for part in ("norm", "layernorm", "rms_norm")):
                groups["norm"].append(name)
            elif any(part in lower for part in ("router", "expert", "moe", "mixture")):
                groups["router"].append(name)
            else:
                groups["other"].append(name)
        return groups

    def summary(self, require_shards: bool = False) -> dict[str, Any]:
        metadata = self.index.get("metadata", {})
        total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
        shard_status = {}
        for shard in self.shard_names:
            try:
                path = self.resolve_shard_path(shard)
                shard_status[shard] = {"present": True, "path": str(path), "bytes": path.stat().st_size}
            except FileNotFoundError:
                shard_status[shard] = {"present": False}
                if require_shards:
                    raise
        return {
            "index": str(self.index_path),
            "tensor_count": len(self.weight_map),
            "shard_count": len(self.shard_names),
            "metadata_total_size": total_size,
            "shards": shard_status,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Gemma safetensors shards without loading all weights.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--shard-root", action="append", default=[])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--tensor", default=None)
    parser.add_argument("--read-bytes", action="store_true")
    parser.add_argument("--require-shards", action="store_true")
    args = parser.parse_args()

    try:
        if args.model_dir:
            weights = GemmaWeightIndex.from_model_dir(args.model_dir)
        elif args.index:
            weights = GemmaWeightIndex(args.index, args.shard_root)
        else:
            raise ValueError("Provide --model-dir or --index")

        result: dict[str, Any] = {"summary": weights.summary(require_shards=args.require_shards)}
        if args.layer is not None:
            result["layer"] = args.layer
            result["groups"] = weights.tensor_groups_for_layer(args.layer)
        if args.tensor:
            ref = weights.tensor_ref(args.tensor, require_shard=args.require_shards or args.read_bytes)
            result["tensor"] = {
                "name": ref.name,
                "shard": ref.shard,
                "dtype": ref.dtype,
                "shape": list(ref.shape),
                "nbytes": ref.nbytes,
                "path": str(ref.path) if ref.path else None,
            }
            if args.read_bytes:
                result["tensor"]["read_bytes"] = len(weights.read_tensor_bytes(args.tensor))

        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"gemma_runtime.weights: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
