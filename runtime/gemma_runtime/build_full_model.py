from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .quantizer import SUPPORTED_DTYPES, QuantizedTensorRecord, quantize_tensor_streaming
from .weights import GemmaWeightIndex


LANGUAGE_PREFIX = "model.language_model."


def tensor_role(name: str) -> str:
    lower = name.lower()
    if ".self_attn." in lower:
        return "attention"
    if ".mlp." in lower:
        return "mlp"
    if "norm" in lower or "layernorm" in lower:
        return "norm"
    if "router" in lower:
        return "router"
    if "embed" in lower:
        return "embedding"
    if "projection" in lower or "laurel" in lower or "altup" in lower:
        return "adapter"
    return "other"


def select_language_tensors(weights: GemmaWeightIndex, include_embeddings: bool) -> list[str]:
    selected = []
    for name in weights.tensor_names:
        if not name.startswith(LANGUAGE_PREFIX):
            continue
        if not include_embeddings and "embed_tokens" in name:
            continue
        selected.append(name)
    return selected


def summarize_records(records: list[QuantizedTensorRecord]) -> dict[str, Any]:
    by_role: dict[str, dict[str, int]] = {}
    for record in records:
        role = tensor_role(record.name)
        entry = by_role.setdefault(role, {"tensor_count": 0, "source_bytes": 0, "quantized_bytes": 0})
        entry["tensor_count"] += 1
        entry["source_bytes"] += record.source_bytes
        entry["quantized_bytes"] += record.quantized_bytes
    return by_role


def write_manifest(
    path: Path,
    model_dir: str | Path,
    output_dir: Path,
    block_size: int,
    records: list[QuantizedTensorRecord],
    elapsed_sec: float,
    complete: bool,
) -> None:
    manifest = {
        "format": "CCQ4_FULL_LANGUAGE_MODEL",
        "version": 1,
        "complete": complete,
        "model_dir": str(Path(model_dir)),
        "output_dir": str(output_dir),
        "block_size": block_size,
        "tensor_count": len(records),
        "source_bytes": sum(record.source_bytes for record in records),
        "quantized_bytes": sum(record.quantized_bytes for record in records),
        "elapsed_sec": round(elapsed_sec, 3),
        "roles": summarize_records(records),
        "tensors": [record.to_json() for record in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a complete CCQ4 Gemma language-model artifact in one resumable run.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="build/gemma_full_ccq4")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument("--max-tensors", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--chunk-blocks", type=int, default=4096)
    args = parser.parse_args()

    start = time.time()
    try:
        if args.block_size <= 0 or args.block_size % 2 != 0:
            raise ValueError("--block-size must be a positive even integer")
        weights = GemmaWeightIndex.from_model_dir(args.model_dir)
        output_dir = Path(args.output_dir)
        manifest_out = Path(args.manifest_out) if args.manifest_out else output_dir / "ccq4_full_language_manifest.json"
        names = select_language_tensors(weights, include_embeddings=args.include_embeddings)
        refs = [weights.tensor_ref(name, require_shard=True) for name in names]
        supported = [ref.name for ref in refs if ref.dtype in SUPPORTED_DTYPES and ref.numel > 0]
        if args.max_tensors is not None:
            supported = supported[:args.max_tensors]

        records: list[QuantizedTensorRecord] = []
        total = len(supported)
        for index, name in enumerate(supported, start=1):
            record = quantize_tensor_streaming(
                weights=weights,
                tensor_name=name,
                output_dir=output_dir,
                block_size=args.block_size,
                overwrite=args.overwrite,
                chunk_blocks=args.chunk_blocks,
            )
            records.append(record)
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == total):
                elapsed = time.time() - start
                print(json.dumps({
                    "progress": f"{index}/{total}",
                    "tensor": name,
                    "source_bytes_done": sum(r.source_bytes for r in records),
                    "quantized_bytes_done": sum(r.quantized_bytes for r in records),
                    "elapsed_sec": round(elapsed, 3),
                }, ensure_ascii=True))
                write_manifest(manifest_out, args.model_dir, output_dir, args.block_size, records, elapsed, complete=False)

        elapsed = time.time() - start
        write_manifest(manifest_out, args.model_dir, output_dir, args.block_size, records, elapsed, complete=True)
        print(json.dumps({
            "wrote": str(manifest_out),
            "tensor_count": len(records),
            "source_bytes": sum(r.source_bytes for r in records),
            "quantized_bytes": sum(r.quantized_bytes for r in records),
            "elapsed_sec": round(elapsed, 3),
        }, indent=2))
        return 0
    except Exception as exc:
        print(f"gemma_runtime.build_full_model: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
