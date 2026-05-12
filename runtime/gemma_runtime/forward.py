from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .matvec import GpuCcq4Session
from .quantizer import Ccq4Tensor, open_ccq4


def gelu_pytorch_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def ccq4_path(root: Path, name: str) -> Path:
    return root / f"{name.replace('/', '_').replace(chr(92), '_').replace(':', '_')}.ccq4"


def read_ccq4_vector(path: str | Path) -> np.ndarray:
    tensor = open_ccq4(path)
    if len(tensor.shape) != 1:
        raise ValueError(f"{path} is not a vector: shape={tensor.shape}")
    return dequantize_ccq4_flat(tensor)


def dequantize_ccq4_flat(tensor: Ccq4Tensor, start_block: int = 0, block_count: int | None = None) -> np.ndarray:
    if tensor.block_size <= 0:
        raise ValueError(f"{tensor.path} has invalid block size")
    total_blocks = tensor.scale_count
    if block_count is None:
        block_count = total_blocks - start_block
    block_count = min(block_count, total_blocks - start_block)
    packed_block_bytes = (tensor.block_size + 1) // 2
    with tensor.path.open("rb") as f:
        f.seek(tensor.scales_offset + start_block * 4)
        scales = np.frombuffer(f.read(block_count * 4), dtype=np.float32).copy()
        f.seek(tensor.data_offset + start_block * packed_block_bytes)
        packed = np.frombuffer(f.read(block_count * packed_block_bytes), dtype=np.uint8).copy()
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    q = np.empty(packed.size * 2, dtype=np.int8)
    q[0::2] = lo
    q[1::2] = hi
    values = q.reshape(block_count, tensor.block_size).astype(np.float32) * scales[:, None]
    start_value = start_block * tensor.block_size
    remaining = tensor.numel - start_value
    return values.reshape(-1)[: min(block_count * tensor.block_size, remaining)]


def dequantize_ccq4_row(path: str | Path, row: int) -> np.ndarray:
    tensor = open_ccq4(path)
    if len(tensor.shape) != 2:
        raise ValueError(f"{path} is not a matrix: shape={tensor.shape}")
    rows, cols = tensor.shape
    if row < 0 or row >= rows:
        raise ValueError(f"row {row} outside 0..{rows - 1}")
    if cols % tensor.block_size != 0:
        full = dequantize_ccq4_flat(tensor)
        return full[row * cols:(row + 1) * cols]
    blocks_per_row = cols // tensor.block_size
    return dequantize_ccq4_flat(tensor, row * blocks_per_row, blocks_per_row)


def dequantize_ccq4_rows(path: str | Path, start_row: int, row_count: int) -> np.ndarray:
    tensor = open_ccq4(path)
    if len(tensor.shape) != 2:
        raise ValueError(f"{path} is not a matrix: shape={tensor.shape}")
    rows, cols = tensor.shape
    row_count = max(0, min(row_count, rows - start_row))
    if cols % tensor.block_size != 0:
        full = dequantize_ccq4_flat(tensor).reshape(rows, cols)
        return full[start_row:start_row + row_count]
    blocks_per_row = cols // tensor.block_size
    flat = dequantize_ccq4_flat(tensor, start_row * blocks_per_row, row_count * blocks_per_row)
    return flat.reshape(row_count, cols)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    mean_square = np.mean(x * x, axis=-1, keepdims=True)
    scale = 1.0 / np.sqrt(mean_square + np.float32(eps))
    return (x * scale * weight).astype(np.float32)


def apply_rope(vec: np.ndarray, position: int, head_dim: int, theta: float) -> np.ndarray:
    out = vec.copy().reshape(-1, head_dim)
    half = head_dim // 2
    idx = np.arange(half, dtype=np.float32)
    inv_freq = np.power(np.float32(theta), -idx / np.float32(half))
    angles = np.float32(position) * inv_freq
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    x1 = out[:, :half].copy()
    x2 = out[:, half:].copy()
    out[:, :half] = x1 * cos - x2 * sin
    out[:, half:] = x1 * sin + x2 * cos
    return out.reshape(-1)


def grouped_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    key_cache: list[np.ndarray],
    value_cache: list[np.ndarray],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, list[float]]:
    key_cache.append(k.copy())
    value_cache.append(v.copy())
    keys = np.stack(key_cache, axis=0).reshape(len(key_cache), kv_heads, head_dim)
    values = np.stack(value_cache, axis=0).reshape(len(value_cache), kv_heads, head_dim)
    qv = q.reshape(q_heads, head_dim)
    repeats = q_heads // kv_heads
    context = np.empty((q_heads, head_dim), dtype=np.float32)
    last_scores: list[float] = []
    scale = 1.0 / math.sqrt(float(head_dim))
    for q_head in range(q_heads):
        kv_head = q_head // repeats
        scores = (keys[:, kv_head, :] @ qv[q_head]) * scale
        scores = scores - np.max(scores)
        probs = np.exp(scores).astype(np.float32)
        probs /= np.sum(probs)
        context[q_head] = probs @ values[:, kv_head, :]
        last_scores.append(float(scores[-1]))
    return context.reshape(-1), last_scores


@dataclass
class ForwardConfig:
    hidden_size: int = 2048
    q_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 256
    num_layers: int = 35
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 1_000_000.0
    vocab_size: int = 262400


class GemmaForwardRuntime:
    def __init__(self, ccq4_dir: str | Path, dll: str | Path, gpu: int = 0, config: ForwardConfig | None = None):
        self.ccq4_dir = Path(ccq4_dir)
        self.dll = dll
        self.gpu = gpu
        self.config = config or ForwardConfig()
        self.key_cache: list[list[np.ndarray]] = [[] for _ in range(self.config.num_layers)]
        self.value_cache: list[list[np.ndarray]] = [[] for _ in range(self.config.num_layers)]
        self.norm_cache: dict[str, np.ndarray] = {}

    def tensor_path(self, name: str) -> Path:
        path = ccq4_path(self.ccq4_dir, name)
        if not path.exists():
            raise FileNotFoundError(f"missing CCQ4 tensor: {path}")
        return path

    def norm_weight(self, name: str) -> np.ndarray:
        if name not in self.norm_cache:
            self.norm_cache[name] = read_ccq4_vector(self.tensor_path(name))
        return self.norm_cache[name]

    def gpu_matvec(self, session: GpuCcq4Session, name: str, x: np.ndarray, resident: bool = True) -> np.ndarray:
        path = self.tensor_path(name)
        values = x.astype(np.float32).tolist()
        if resident:
            return np.asarray(session.matvec_resident(path, values), dtype=np.float32)
        return np.asarray(session.matvec(path, values), dtype=np.float32)

    def embedding(self, token_id: int) -> np.ndarray:
        return dequantize_ccq4_row(self.tensor_path("model.language_model.embed_tokens.weight"), token_id).astype(np.float32)

    def layer(self, session: GpuCcq4Session, x: np.ndarray, layer_index: int, position: int, run_mlp: bool, resident: bool) -> dict[str, Any]:
        cfg = self.config
        prefix = f"model.language_model.layers.{layer_index}"
        residual = x
        h = rms_norm(x, self.norm_weight(f"{prefix}.input_layernorm.weight"), cfg.rms_norm_eps)
        q = self.gpu_matvec(session, f"{prefix}.self_attn.q_proj.weight", h, resident=resident)
        k = self.gpu_matvec(session, f"{prefix}.self_attn.k_proj.weight", h, resident=resident)
        v = self.gpu_matvec(session, f"{prefix}.self_attn.v_proj.weight", h, resident=resident)
        q = rms_norm(q.reshape(cfg.q_heads, cfg.head_dim), self.norm_weight(f"{prefix}.self_attn.q_norm.weight"), cfg.rms_norm_eps).reshape(-1)
        k = rms_norm(k.reshape(cfg.kv_heads, cfg.head_dim), self.norm_weight(f"{prefix}.self_attn.k_norm.weight"), cfg.rms_norm_eps).reshape(-1)
        q = apply_rope(q, position, cfg.head_dim, cfg.rope_theta)
        k = apply_rope(k, position, cfg.head_dim, cfg.rope_theta)
        context, scores = grouped_attention(
            q, k, v,
            self.key_cache[layer_index],
            self.value_cache[layer_index],
            cfg.q_heads,
            cfg.kv_heads,
            cfg.head_dim,
        )
        attn_out = self.gpu_matvec(session, f"{prefix}.self_attn.o_proj.weight", context, resident=resident)
        x = residual + attn_out

        mlp_summary = None
        if run_mlp:
            residual = x
            h = rms_norm(x, self.norm_weight(f"{prefix}.pre_feedforward_layernorm.weight"), cfg.rms_norm_eps)
            gate = self.gpu_matvec(session, f"{prefix}.mlp.gate_proj.weight", h, resident=resident)
            up = self.gpu_matvec(session, f"{prefix}.mlp.up_proj.weight", h, resident=resident)
            ff = gelu_pytorch_tanh(gate) * up
            down = self.gpu_matvec(session, f"{prefix}.mlp.down_proj.weight", ff, resident=resident)
            x = residual + down
            mlp_summary = {"intermediate": int(ff.size), "down_norm": float(np.linalg.norm(down))}

        return {
            "hidden": x.astype(np.float32),
            "attention": {
                "q_norm": float(np.linalg.norm(q)),
                "k_norm": float(np.linalg.norm(k)),
                "v_norm": float(np.linalg.norm(v)),
                "out_norm": float(np.linalg.norm(attn_out)),
                "last_scores": scores,
                "cache_len": len(self.key_cache[layer_index]),
            },
            "mlp": mlp_summary,
        }

    def logits_topk(self, hidden: np.ndarray, top_k: int, vocab_limit: int | None = None, chunk_rows: int = 1024) -> list[dict[str, float | int]]:
        final = rms_norm(hidden, self.norm_weight("model.language_model.norm.weight"), self.config.rms_norm_eps)
        embed_path = self.tensor_path("model.language_model.embed_tokens.weight")
        tensor = open_ccq4(embed_path)
        rows, cols = tensor.shape
        if cols != final.size:
            raise ValueError(f"embedding cols {cols} do not match hidden size {final.size}")
        limit = min(vocab_limit or rows, rows)
        best_ids = np.empty(0, dtype=np.int64)
        best_vals = np.empty(0, dtype=np.float32)
        for start in range(0, limit, chunk_rows):
            matrix = dequantize_ccq4_rows(embed_path, start, min(chunk_rows, limit - start))
            logits = matrix @ final
            ids = np.arange(start, start + logits.size, dtype=np.int64)
            all_ids = np.concatenate([best_ids, ids])
            all_vals = np.concatenate([best_vals, logits.astype(np.float32)])
            keep = np.argpartition(-all_vals, min(top_k, all_vals.size) - 1)[:top_k]
            best_ids = all_ids[keep]
            best_vals = all_vals[keep]
        order = np.argsort(-best_vals)
        return [{"token_id": int(best_ids[i]), "logit": float(best_vals[i])} for i in order]

    def forward_token(
        self,
        token_id: int,
        position: int,
        max_layers: int | None = None,
        run_mlp: bool = True,
        top_k: int = 8,
        vocab_limit: int | None = None,
        resident: bool = True,
    ) -> dict[str, Any]:
        cfg = self.config
        layer_count = min(max_layers or cfg.num_layers, cfg.num_layers)
        x = self.embedding(token_id)
        layer_summaries = []
        start = time.time()
        with GpuCcq4Session(self.dll, self.gpu) as session:
            for layer_index in range(layer_count):
                result = self.layer(session, x, layer_index, position, run_mlp, resident=resident)
                x = result["hidden"]
                layer_summaries.append({
                    "layer": layer_index,
                    "hidden_norm": float(np.linalg.norm(x)),
                    "attention": result["attention"],
                    "mlp": result["mlp"],
                })
            resident_count = session.resident_count
        logits = self.logits_topk(x, top_k=top_k, vocab_limit=vocab_limit)
        return {
            "token_id": token_id,
            "position": position,
            "layers_executed": layer_count,
            "run_mlp": run_mlp,
            "resident_weights": resident,
            "resident_matrix_count": resident_count,
            "hidden_norm": float(np.linalg.norm(x)),
            "elapsed_sec": round(time.time() - start, 3),
            "top_logits": logits,
            "layers": layer_summaries,
        }

    def forward_sequence(
        self,
        token_ids: list[int],
        start_position: int = 0,
        max_layers: int | None = None,
        run_mlp: bool = True,
        top_k: int = 8,
        vocab_limit: int | None = None,
        resident: bool = True,
    ) -> dict[str, Any]:
        cfg = self.config
        layer_count = min(max_layers or cfg.num_layers, cfg.num_layers)
        steps = []
        start = time.time()
        with GpuCcq4Session(self.dll, self.gpu) as session:
            for offset, token_id in enumerate(token_ids):
                position = start_position + offset
                x = self.embedding(token_id)
                layer_summaries = []
                for layer_index in range(layer_count):
                    result = self.layer(session, x, layer_index, position, run_mlp, resident=resident)
                    x = result["hidden"]
                    layer_summaries.append({
                        "layer": layer_index,
                        "hidden_norm": float(np.linalg.norm(x)),
                        "cache_len": result["attention"]["cache_len"],
                        "mlp": result["mlp"],
                    })
                logits = self.logits_topk(x, top_k=top_k, vocab_limit=vocab_limit)
                steps.append({
                    "token_id": token_id,
                    "position": position,
                    "hidden_norm": float(np.linalg.norm(x)),
                    "top_logits": logits,
                    "layers": layer_summaries,
                })
            resident_count = session.resident_count
        return {
            "tokens": token_ids,
            "start_position": start_position,
            "layers_executed": layer_count,
            "run_mlp": run_mlp,
            "resident_weights": resident,
            "resident_matrix_count": resident_count,
            "elapsed_sec": round(time.time() - start, 3),
            "steps": steps,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-token Gemma CCQ4 forward loop.")
    parser.add_argument("--ccq4-dir", default="build/gemma_full_ccq4")
    parser.add_argument("--dll", default="build/CC_OpenCl.dll")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--token-id", type=int, default=None)
    parser.add_argument("--tokens", default=None)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=1)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--vocab-limit", type=int, default=2048)
    parser.add_argument("--no-resident-weights", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    try:
        runtime = GemmaForwardRuntime(args.ccq4_dir, args.dll, args.gpu)
        if args.tokens:
            token_ids = [int(part.strip()) for part in args.tokens.split(",") if part.strip()]
            result = runtime.forward_sequence(
                token_ids=token_ids,
                start_position=args.position,
                max_layers=args.max_layers,
                run_mlp=not args.skip_mlp,
                top_k=args.top_k,
                vocab_limit=args.vocab_limit,
                resident=not args.no_resident_weights,
            )
            summary = {
                "tokens": result["tokens"],
                "layers_executed": result["layers_executed"],
                "last_top_logits": result["steps"][-1]["top_logits"] if result["steps"] else [],
                "json_out": args.json_out,
            }
        else:
            if args.token_id is None:
                raise ValueError("Provide --token-id or --tokens")
            result = runtime.forward_token(
                token_id=args.token_id,
                position=args.position,
                max_layers=args.max_layers,
                run_mlp=not args.skip_mlp,
                top_k=args.top_k,
                vocab_limit=args.vocab_limit,
                resident=not args.no_resident_weights,
            )
            summary = {
                "token_id": result["token_id"],
                "layers_executed": result["layers_executed"],
                "hidden_norm": result["hidden_norm"],
                "top_logits": result["top_logits"],
                "json_out": args.json_out,
            }
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"gemma_runtime.forward: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
