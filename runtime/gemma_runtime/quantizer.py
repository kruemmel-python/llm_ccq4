from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .weights import DTYPE_BYTES, GemmaWeightIndex, TensorRef

try:
    import numpy as np
except Exception:  # pragma: no cover - exercised when numpy is unavailable.
    np = None


MAGIC = b"CCQ4\x01\x00\x00\x00"
SUPPORTED_DTYPES = {"F32", "F16", "BF16"}


@dataclass(frozen=True)
class QuantizedTensorRecord:
    name: str
    source_shard: str
    dtype: str
    shape: tuple[int, ...]
    block_size: int
    source_bytes: int
    quantized_bytes: int
    max_abs_error: float
    mean_abs_error: float
    output: str

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_shard": self.source_shard,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "block_size": self.block_size,
            "source_bytes": self.source_bytes,
            "quantized_bytes": self.quantized_bytes,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "output": self.output,
        }


@dataclass(frozen=True)
class Ccq4Tensor:
    path: Path
    header: dict[str, Any]
    scales_offset: int
    data_offset: int

    @property
    def name(self) -> str:
        return str(self.header["name"])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.header.get("shape", []))

    @property
    def numel(self) -> int:
        return int(self.header.get("numel", 0))

    @property
    def block_size(self) -> int:
        return int(self.header.get("block_size", 0))

    @property
    def scale_count(self) -> int:
        return (self.numel + self.block_size - 1) // self.block_size


def open_ccq4(path: str | Path) -> Ccq4Tensor:
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{path} is not a CCQ4 file")
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} has truncated CCQ4 header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path} has truncated CCQ4 header")
    header = json.loads(header_bytes.decode("utf-8"))
    if header.get("format") != "CCQ4":
        raise ValueError(f"{path} has unsupported format {header.get('format')}")
    scales_offset = len(MAGIC) + 8 + int(header_len)
    data_offset = scales_offset + int(header["scale_bytes"])
    return Ccq4Tensor(path=path, header=header, scales_offset=scales_offset, data_offset=data_offset)


def dequantize_ccq4_blocks(path: str | Path, start_block: int = 0, block_count: int | None = None) -> list[float]:
    tensor = open_ccq4(path)
    if tensor.block_size <= 0:
        raise ValueError(f"{path} has invalid block_size")
    total_blocks = tensor.scale_count
    if start_block < 0 or start_block >= total_blocks:
        raise ValueError(f"start_block out of range: {start_block}")
    if block_count is None:
        block_count = total_blocks - start_block
    block_count = min(block_count, total_blocks - start_block)
    values: list[float] = []
    packed_block_bytes = (tensor.block_size + 1) // 2
    with tensor.path.open("rb") as f:
        f.seek(tensor.scales_offset + start_block * 4)
        scales = struct.unpack(f"<{block_count}f", f.read(block_count * 4))
        f.seek(tensor.data_offset + start_block * packed_block_bytes)
        for block_index, scale in enumerate(scales):
            raw = f.read(packed_block_bytes)
            remaining = tensor.numel - (start_block + block_index) * tensor.block_size
            count = min(tensor.block_size, remaining)
            q_values = unpack_int4_signed(raw, count)
            values.extend(q * scale for q in q_values)
    return values


def dequantize_ccq4(path: str | Path) -> list[float]:
    return dequantize_ccq4_blocks(path)


def bf16_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def f16_to_float(bits: int) -> float:
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    if exp == 0:
        value = (frac / 1024.0) * (2.0 ** -14) if frac else 0.0
    elif exp == 0x1F:
        value = math.inf if frac == 0 else math.nan
    else:
        value = (1.0 + frac / 1024.0) * (2.0 ** (exp - 15))
    return -value if sign else value


def bytes_to_float_block(data: bytes, dtype: str) -> list[float]:
    if dtype == "F32":
        return list(struct.unpack(f"<{len(data) // 4}f", data))
    if dtype == "BF16":
        values = struct.unpack(f"<{len(data) // 2}H", data)
        return [bf16_to_float(v) for v in values]
    if dtype == "F16":
        values = struct.unpack(f"<{len(data) // 2}H", data)
        return [f16_to_float(v) for v in values]
    raise ValueError(f"unsupported dtype for quantization: {dtype}")


def pack_int4_signed(values: list[int]) -> bytes:
    out = bytearray((len(values) + 1) // 2)
    for i, value in enumerate(values):
        nibble = value & 0x0F
        if i & 1:
            out[i // 2] |= nibble << 4
        else:
            out[i // 2] |= nibble
    return bytes(out)


def unpack_int4_signed(data: bytes, count: int) -> list[int]:
    values: list[int] = []
    for byte in data:
        lo = byte & 0x0F
        hi = (byte >> 4) & 0x0F
        values.append(lo - 16 if lo >= 8 else lo)
        if len(values) == count:
            break
        values.append(hi - 16 if hi >= 8 else hi)
        if len(values) == count:
            break
    return values


def quantize_float_values_int4(values: list[float], block_size: int) -> tuple[bytes, bytes, float, float]:
    scales = bytearray()
    packed_blocks = bytearray()
    max_abs_error = 0.0
    sum_abs_error = 0.0
    count = 0

    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        max_abs = max((abs(v) for v in block if math.isfinite(v)), default=0.0)
        scale = max_abs / 7.0 if max_abs > 0.0 else 1.0
        q_values: list[int] = []
        for value in block:
            finite = value if math.isfinite(value) else 0.0
            q = int(round(finite / scale)) if scale > 0.0 else 0
            q = max(-8, min(7, q))
            q_values.append(q)
            recon = q * scale
            err = abs(finite - recon)
            max_abs_error = max(max_abs_error, err)
            sum_abs_error += err
            count += 1
        scales.extend(struct.pack("<f", scale))
        packed_blocks.extend(pack_int4_signed(q_values))

    mean_abs_error = sum_abs_error / max(count, 1)
    return bytes(scales), bytes(packed_blocks), max_abs_error, mean_abs_error


def quantize_float_block_int4(values: list[float], block_size: int) -> tuple[bytes, bytes, float, float, int]:
    if len(values) > block_size:
        raise ValueError(f"block has {len(values)} values, expected <= {block_size}")
    max_abs = max((abs(v) for v in values if math.isfinite(v)), default=0.0)
    scale = max_abs / 7.0 if max_abs > 0.0 else 1.0
    q_values: list[int] = []
    max_abs_error = 0.0
    sum_abs_error = 0.0
    for value in values:
        finite = value if math.isfinite(value) else 0.0
        q = int(round(finite / scale)) if scale > 0.0 else 0
        q = max(-8, min(7, q))
        q_values.append(q)
        err = abs(finite - q * scale)
        max_abs_error = max(max_abs_error, err)
        sum_abs_error += err
    return struct.pack("<f", scale), pack_int4_signed(q_values), max_abs_error, sum_abs_error, len(values)


def quantize_raw_bytes_int4_fast(raw: bytes, dtype: str, value_count: int, block_size: int) -> tuple[bytes, bytes, float, float, int] | None:
    if np is None:
        return None
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16, count=value_count).astype(np.uint32) << 16
        values = bits.view(np.float32)
    elif dtype == "F16":
        values = np.frombuffer(raw, dtype=np.float16, count=value_count).astype(np.float32)
    elif dtype == "F32":
        values = np.frombuffer(raw, dtype=np.float32, count=value_count)
    else:
        return None

    block_count = (value_count + block_size - 1) // block_size
    padded_count = block_count * block_size
    if padded_count != value_count:
        padded = np.zeros(padded_count, dtype=np.float32)
        padded[:value_count] = values
        values_2d = padded.reshape(block_count, block_size)
        valid = np.zeros(padded_count, dtype=bool)
        valid[:value_count] = True
        valid_2d = valid.reshape(block_count, block_size)
    else:
        values_2d = values.reshape(block_count, block_size)
        valid_2d = None

    finite = np.where(np.isfinite(values_2d), values_2d, 0.0)
    abs_values = np.abs(finite)
    if valid_2d is not None:
        abs_values = np.where(valid_2d, abs_values, 0.0)
    max_abs = np.max(abs_values, axis=1).astype(np.float32)
    scales = np.where(max_abs > 0.0, max_abs / np.float32(7.0), np.float32(1.0)).astype(np.float32)
    q = np.rint(finite / scales[:, None]).clip(-8, 7).astype(np.int8)
    if valid_2d is not None:
        q = np.where(valid_2d, q, 0).astype(np.int8)
    recon = q.astype(np.float32) * scales[:, None]
    err = np.abs(finite - recon)
    if valid_2d is not None:
        err = np.where(valid_2d, err, 0.0)
    q_nibbles = (q.astype(np.int16) & 0x0F).astype(np.uint8)
    packed = (q_nibbles[:, 0::2] | (q_nibbles[:, 1::2] << 4)).reshape(-1)
    return (
        scales.tobytes(),
        packed.tobytes(),
        float(np.max(err)) if value_count else 0.0,
        float(np.sum(err)),
        value_count,
    )


def read_tensor_float_values(weights: GemmaWeightIndex, ref: TensorRef) -> list[float]:
    if ref.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"{ref.name}: dtype {ref.dtype} is not supported yet")
    raw = weights.read_tensor_bytes(ref.name)
    expected = ref.nbytes
    if expected is not None and len(raw) != expected:
        raise ValueError(f"{ref.name}: read {len(raw)} bytes, expected {expected}")
    return bytes_to_float_block(raw, ref.dtype)


def ccq4_payload_sizes(numel: int, block_size: int) -> tuple[int, int, int]:
    block_count = (numel + block_size - 1) // block_size
    scale_bytes = block_count * 4
    data_bytes = block_count * ((block_size + 1) // 2)
    return block_count, scale_bytes, data_bytes


def ccq4_header_bytes(ref: TensorRef, block_size: int, scale_bytes: int, data_bytes: int) -> bytes:
    header = {
        "format": "CCQ4",
        "version": 1,
        "name": ref.name,
        "source_shard": ref.shard,
        "source_dtype": ref.dtype,
        "shape": list(ref.shape),
        "block_size": block_size,
        "numel": ref.numel,
        "scale_dtype": "F32",
        "quant_dtype": "INT4_SIGNED",
        "scale_bytes": scale_bytes,
        "data_bytes": data_bytes,
    }
    return json.dumps(header, separators=(",", ":")).encode("utf-8")


def write_ccq4(path: Path, ref: TensorRef, block_size: int, scales: bytes, packed: bytes) -> int:
    header_bytes = ccq4_header_bytes(ref, block_size, len(scales), len(packed))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(scales)
        f.write(packed)
    return path.stat().st_size


def quantized_path_for_name(output_dir: Path, tensor_name: str) -> Path:
    safe_name = tensor_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return output_dir / f"{safe_name}.ccq4"


def quantize_tensor(
    weights: GemmaWeightIndex,
    tensor_name: str,
    output_dir: Path,
    block_size: int,
) -> QuantizedTensorRecord:
    ref = weights.tensor_ref(tensor_name, require_shard=True)
    values = read_tensor_float_values(weights, ref)
    scales, packed, max_err, mean_err = quantize_float_values_int4(values, block_size)
    out_path = quantized_path_for_name(output_dir, tensor_name)
    quantized_bytes = write_ccq4(out_path, ref, block_size, scales, packed)
    return QuantizedTensorRecord(
        name=tensor_name,
        source_shard=ref.shard,
        dtype=ref.dtype or "unknown",
        shape=ref.shape,
        block_size=block_size,
        source_bytes=ref.nbytes or 0,
        quantized_bytes=quantized_bytes,
        max_abs_error=round(max_err, 8),
        mean_abs_error=round(mean_err, 8),
        output=str(out_path),
    )


def quantize_tensor_streaming(
    weights: GemmaWeightIndex,
    tensor_name: str,
    output_dir: Path,
    block_size: int,
    overwrite: bool = False,
    chunk_blocks: int = 4096,
) -> QuantizedTensorRecord:
    ref = weights.tensor_ref(tensor_name, require_shard=True)
    if ref.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"{ref.name}: dtype {ref.dtype} is not supported yet")
    if ref.nbytes is None or ref.absolute_offsets is None or ref.path is None:
        raise ValueError(f"{ref.name}: missing tensor offsets")
    if chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be positive")
    out_path = quantized_path_for_name(output_dir, tensor_name)
    if out_path.exists() and not overwrite:
        existing = open_ccq4(out_path)
        if existing.name == tensor_name and existing.numel == ref.numel and existing.block_size == block_size:
            return QuantizedTensorRecord(
                name=tensor_name,
                source_shard=ref.shard,
                dtype=ref.dtype or "unknown",
                shape=ref.shape,
                block_size=block_size,
                source_bytes=ref.nbytes or 0,
                quantized_bytes=out_path.stat().st_size,
                max_abs_error=0.0,
                mean_abs_error=0.0,
                output=str(out_path),
            )

    dtype_bytes = DTYPE_BYTES[ref.dtype]
    block_count, scale_bytes, data_bytes = ccq4_payload_sizes(ref.numel, block_size)
    header_bytes = ccq4_header_bytes(ref, block_size, scale_bytes, data_bytes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    max_abs_error = 0.0
    sum_abs_error = 0.0
    count = 0
    start, _ = ref.absolute_offsets

    with ref.path.open("rb") as src, tmp_path.open("wb") as dst:
        dst.write(MAGIC)
        dst.write(struct.pack("<Q", len(header_bytes)))
        dst.write(header_bytes)
        scales_pos = dst.tell()
        dst.write(b"\x00" * scale_bytes)
        src.seek(start)
        scales = bytearray()
        block_index = 0
        while block_index < block_count:
            blocks_now = min(chunk_blocks, block_count - block_index)
            remaining = ref.numel - block_index * block_size
            value_count = min(blocks_now * block_size, remaining)
            raw = src.read(value_count * dtype_bytes)
            if len(raw) != value_count * dtype_bytes:
                raise ValueError(f"{ref.name}: truncated source while reading block {block_index}")
            fast = quantize_raw_bytes_int4_fast(raw, ref.dtype, value_count, block_size)
            if fast is None:
                values = bytes_to_float_block(raw, ref.dtype)
                scale, packed, chunk_max, chunk_mean = quantize_float_values_int4(values, block_size)
                chunk_sum = chunk_mean * len(values)
                chunk_count = len(values)
            else:
                scale, packed, chunk_max, chunk_sum, chunk_count = fast
            scales.extend(scale)
            dst.write(packed)
            max_abs_error = max(max_abs_error, chunk_max)
            sum_abs_error += chunk_sum
            count += chunk_count
            block_index += blocks_now
        if len(scales) != scale_bytes:
            raise ValueError(f"{ref.name}: wrote {len(scales)} scale bytes, expected {scale_bytes}")
        dst.seek(scales_pos)
        dst.write(scales)

    tmp_path.replace(out_path)
    return QuantizedTensorRecord(
        name=tensor_name,
        source_shard=ref.shard,
        dtype=ref.dtype or "unknown",
        shape=ref.shape,
        block_size=block_size,
        source_bytes=ref.nbytes or 0,
        quantized_bytes=out_path.stat().st_size,
        max_abs_error=round(max_abs_error, 8),
        mean_abs_error=round(sum_abs_error / max(count, 1), 8),
        output=str(out_path),
    )


def select_tensors(weights: GemmaWeightIndex, layer: int | None, group: str | None, tensor: str | None) -> list[str]:
    if tensor:
        return [tensor]
    if layer is None:
        raise ValueError("Provide --tensor or --layer")
    groups = weights.tensor_groups_for_layer(layer)
    if group:
        if group not in groups:
            raise ValueError(f"Unknown group {group}; valid groups: {', '.join(groups)}")
        return groups[group]
    names: list[str] = []
    for key in ("attention", "mlp", "norm", "router", "other"):
        names.extend(groups[key])
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Blockwise INT4 quantizer for Gemma safetensors tensors.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default="build/gemma_quantized")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--group", choices=["attention", "mlp", "norm", "router", "other"], default=None)
    parser.add_argument("--tensor", default=None)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-tensors", type=int, default=None)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect-ccq4", default=None)
    parser.add_argument("--dequantize-ccq4", default=None)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--block-count", type=int, default=None)
    args = parser.parse_args()

    try:
        if args.inspect_ccq4:
            tensor = open_ccq4(args.inspect_ccq4)
            print(json.dumps({
                "path": str(tensor.path),
                "name": tensor.name,
                "shape": list(tensor.shape),
                "numel": tensor.numel,
                "block_size": tensor.block_size,
                "scale_count": tensor.scale_count,
                "header": tensor.header,
            }, indent=2))
            return 0
        if args.dequantize_ccq4:
            values = dequantize_ccq4_blocks(args.dequantize_ccq4, args.start_block, args.block_count)
            if values:
                max_abs = max(abs(v) for v in values)
                mean_abs = sum(abs(v) for v in values) / len(values)
            else:
                max_abs = 0.0
                mean_abs = 0.0
            print(json.dumps({
                "path": args.dequantize_ccq4,
                "value_count": len(values),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "first_values": values[:16],
            }, indent=2))
            return 0

        if args.block_size <= 0 or args.block_size % 2 != 0:
            raise ValueError("--block-size must be a positive even integer")
        if not args.model_dir:
            raise ValueError("--model-dir is required for quantization mode")
        weights = GemmaWeightIndex.from_model_dir(args.model_dir)
        selected = select_tensors(weights, args.layer, args.group, args.tensor)
        if args.max_tensors is not None:
            selected = selected[:args.max_tensors]
        selected_refs = [weights.tensor_ref(name, require_shard=not args.dry_run) for name in selected]
        output_dir = Path(args.output_dir)

        if args.dry_run:
            manifest = {
                "mode": "dry_run",
                "tensor_count": len(selected_refs),
                "tensors": [
                    {
                        "name": ref.name,
                        "shard": ref.shard,
                        "dtype": ref.dtype,
                        "shape": list(ref.shape),
                        "nbytes": ref.nbytes,
                    }
                    for ref in selected_refs
                ],
            }
        else:
            records = [
                quantize_tensor_streaming(weights, ref.name, output_dir, args.block_size, overwrite=args.overwrite)
                if args.streaming else
                quantize_tensor(weights, ref.name, output_dir, args.block_size)
                for ref in selected_refs
            ]
            manifest = {
                "format": "CCQ4_MANIFEST",
                "version": 1,
                "model_dir": str(Path(args.model_dir)),
                "block_size": args.block_size,
                "tensor_count": len(records),
                "source_bytes": sum(record.source_bytes for record in records),
                "quantized_bytes": sum(record.quantized_bytes for record in records),
                "tensors": [record.to_json() for record in records],
            }

        if args.manifest_out:
            out = Path(args.manifest_out)
        else:
            suffix = "dry_run" if args.dry_run else "manifest"
            target = args.tensor.replace(".", "_") if args.tensor else f"layer_{args.layer}_{args.group or 'all'}"
            out = output_dir / f"{target}.{suffix}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "wrote": str(out),
            "tensor_count": manifest["tensor_count"],
            "source_bytes": manifest.get("source_bytes"),
            "quantized_bytes": manifest.get("quantized_bytes"),
        }, indent=2))
        return 0
    except Exception as exc:
        print(f"gemma_runtime.quantizer: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
