from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .matvec import GpuCcq4Session, ccq4_matvec, deterministic_vector, read_f32_vector, write_f32_vector
from .quantizer import quantize_tensor
from .weights import GemmaWeightIndex


PROJECTION_SUFFIXES = {
    "q": ".self_attn.q_proj.weight",
    "k": ".self_attn.k_proj.weight",
    "v": ".self_attn.v_proj.weight",
    "o": ".self_attn.o_proj.weight",
}


def attention_projection_tensors(weights: GemmaWeightIndex, layer: int) -> dict[str, str]:
    groups = weights.tensor_groups_for_layer(layer)
    attention = groups["attention"]
    selected: dict[str, str] = {}
    for key, suffix in PROJECTION_SUFFIXES.items():
        matches = [name for name in attention if name.endswith(suffix)]
        if matches:
            selected[key] = matches[0]
    return selected


def quantized_path_for_tensor(output_dir: Path, tensor_name: str) -> Path:
    safe_name = tensor_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return output_dir / f"{safe_name}.ccq4"


def ensure_projection_quantized(
    weights: GemmaWeightIndex,
    layer: int,
    output_dir: Path,
    block_size: int,
    projections: list[str],
) -> dict[str, Path]:
    tensors = attention_projection_tensors(weights, layer)
    outputs: dict[str, Path] = {}
    for projection in projections:
        tensor_name = tensors.get(projection)
        if not tensor_name:
            raise KeyError(f"projection {projection!r} not found for layer {layer}")
        out_path = quantized_path_for_tensor(output_dir, tensor_name)
        if not out_path.exists():
            quantize_tensor(weights, tensor_name, output_dir, block_size)
        outputs[projection] = out_path
    return outputs


def run_projection_runtime(
    model_dir: str | Path,
    layer: int,
    output_dir: str | Path,
    dll: str | Path,
    gpu: int,
    block_size: int,
    projections: list[str],
    seed: int,
    compare_cpu: bool,
    gpu_session: GpuCcq4Session | None = None,
) -> dict:
    if gpu_session is None:
        with GpuCcq4Session(dll, gpu) as session:
            return run_projection_runtime(
                model_dir=model_dir,
                layer=layer,
                output_dir=output_dir,
                dll=dll,
                gpu=gpu,
                block_size=block_size,
                projections=projections,
                seed=seed,
                compare_cpu=compare_cpu,
                gpu_session=session,
            )

    output_dir = Path(output_dir)
    weights = GemmaWeightIndex.from_model_dir(model_dir)
    ccq4_paths = ensure_projection_quantized(weights, layer, output_dir, block_size, projections)

    # Gemma text hidden size is 2048 for this model; derive it from the first projection matrix.
    from .quantizer import open_ccq4

    first = open_ccq4(next(iter(ccq4_paths.values())))
    if len(first.shape) != 2:
        raise ValueError(f"projection tensor must be rank-2, got {first.shape}")
    hidden_size = first.shape[1]
    x = deterministic_vector(hidden_size, seed=seed)
    input_path = output_dir / f"layer{layer}_projection_input.f32"
    write_f32_vector(input_path, x)

    records = []
    for projection, ccq4_path in ccq4_paths.items():
        y_gpu = gpu_session.matvec(ccq4_path, x)
        out_path = output_dir / f"layer{layer}_{projection}_proj_gpu.f32"
        write_f32_vector(out_path, y_gpu)
        record = {
            "projection": projection,
            "ccq4": str(ccq4_path),
            "output": str(out_path),
            "output_count": len(y_gpu),
            "gpu_first_values": y_gpu[:8],
        }
        if compare_cpu:
            y_cpu = ccq4_matvec(ccq4_path, x)
            record["max_abs_error_vs_cpu"] = max(abs(a - b) for a, b in zip(y_gpu, y_cpu)) if y_cpu else 0.0
        records.append(record)

    return {
        "model_dir": str(model_dir),
        "layer": layer,
        "block_size": block_size,
        "hidden_size": hidden_size,
        "input": str(input_path),
        "projections": records,
    }


def repeat_kv_to_q_heads(values: list[float], q_heads: int, kv_heads: int, head_dim: int) -> list[float]:
    if kv_heads <= 0 or q_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError("q_heads must be a positive multiple of kv_heads")
    if len(values) != kv_heads * head_dim:
        raise ValueError(f"KV vector length {len(values)} does not match kv_heads*head_dim")
    repeats = q_heads // kv_heads
    out: list[float] = []
    for kv_head in range(kv_heads):
        start = kv_head * head_dim
        block = values[start:start + head_dim]
        for _ in range(repeats):
            out.extend(block)
    return out


def single_token_attention_context(
    q: list[float],
    k: list[float],
    v: list[float],
    q_heads: int = 8,
    kv_heads: int = 2,
    head_dim: int = 256,
) -> dict:
    expected_q = q_heads * head_dim
    expected_kv = kv_heads * head_dim
    if len(q) != expected_q:
        raise ValueError(f"Q length {len(q)} does not match {expected_q}")
    if len(k) != expected_kv or len(v) != expected_kv:
        raise ValueError(f"K/V lengths {len(k)}/{len(v)} do not match {expected_kv}")

    expanded_k = repeat_kv_to_q_heads(k, q_heads, kv_heads, head_dim)
    expanded_v = repeat_kv_to_q_heads(v, q_heads, kv_heads, head_dim)
    scores = []
    context = []
    scale = 1.0 / math.sqrt(float(head_dim))
    for head in range(q_heads):
        start = head * head_dim
        q_head = q[start:start + head_dim]
        k_head = expanded_k[start:start + head_dim]
        v_head = expanded_v[start:start + head_dim]
        # Single-token self-attention softmax is 1.0; keep score for diagnostics.
        scores.append(sum(a * b for a, b in zip(q_head, k_head)) * scale)
        context.extend(v_head)
    return {
        "context": context,
        "scores": scores,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
    }


def run_attention_core_runtime(
    model_dir: str | Path,
    layer: int,
    output_dir: str | Path,
    dll: str | Path,
    gpu: int,
    block_size: int,
    seed: int,
    compare_cpu: bool,
    q_heads: int = 8,
    kv_heads: int = 2,
    head_dim: int = 256,
) -> dict:
    with GpuCcq4Session(dll, gpu) as session:
        projection_result = run_projection_runtime(
            model_dir=model_dir,
            layer=layer,
            output_dir=output_dir,
            dll=dll,
            gpu=gpu,
            block_size=block_size,
            projections=["q", "k", "v", "o"],
            seed=seed,
            compare_cpu=compare_cpu,
            gpu_session=session,
        )
        output_dir = Path(output_dir)
        projection_by_name = {record["projection"]: record for record in projection_result["projections"]}
        q = read_f32_vector(projection_by_name["q"]["output"])
        k = read_f32_vector(projection_by_name["k"]["output"])
        v = read_f32_vector(projection_by_name["v"]["output"])
        attention = single_token_attention_context(q, k, v, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim)
        context = attention["context"]
        context_path = output_dir / f"layer{layer}_attention_context.f32"
        write_f32_vector(context_path, context)

        o_ccq4 = projection_by_name["o"]["ccq4"]
        o_gpu = session.matvec(o_ccq4, context)
    o_path = output_dir / f"layer{layer}_o_proj_from_context_gpu.f32"
    write_f32_vector(o_path, o_gpu)
    o_record = {
        "ccq4": o_ccq4,
        "output": str(o_path),
        "output_count": len(o_gpu),
        "gpu_first_values": o_gpu[:8],
    }
    if compare_cpu:
        o_cpu = ccq4_matvec(o_ccq4, context)
        o_record["max_abs_error_vs_cpu"] = max(abs(a - b) for a, b in zip(o_gpu, o_cpu)) if o_cpu else 0.0

    return {
        "projection_runtime": projection_result,
        "attention": {
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "scores": attention["scores"],
            "context": str(context_path),
            "context_count": len(context),
        },
        "o_projection_from_context": o_record,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gemma layer attention projections through CCQ4 GPU matvec.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output-dir", default="build/gemma_quantized")
    parser.add_argument("--dll", default="build/CC_OpenCl.dll")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--projection", action="append", choices=sorted(PROJECTION_SUFFIXES), default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--attention-core", action="store_true")
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    try:
        if args.attention_core:
            result = run_attention_core_runtime(
                model_dir=args.model_dir,
                layer=args.layer,
                output_dir=args.output_dir,
                dll=args.dll,
                gpu=args.gpu,
                block_size=args.block_size,
                seed=args.seed,
                compare_cpu=args.compare_cpu,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
            )
            summary = {
                "layer": args.layer,
                "attention_context_count": result["attention"]["context_count"],
                "o_output": result["o_projection_from_context"]["output"],
            }
        else:
            projections = args.projection if args.projection else ["q", "k", "v"]
            result = run_projection_runtime(
                model_dir=args.model_dir,
                layer=args.layer,
                output_dir=args.output_dir,
                dll=args.dll,
                gpu=args.gpu,
                block_size=args.block_size,
                projections=projections,
                seed=args.seed,
                compare_cpu=args.compare_cpu,
            )
            summary = {
                "layer": result["layer"],
                "hidden_size": result["hidden_size"],
                "projection_count": len(result["projections"]),
                "outputs": [record["output"] for record in result["projections"]],
            }
        out = Path(args.json_out) if args.json_out else Path(args.output_dir) / f"layer{args.layer}_attention_projection_runtime.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summary["wrote"] = str(out)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"gemma_runtime.projection: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
