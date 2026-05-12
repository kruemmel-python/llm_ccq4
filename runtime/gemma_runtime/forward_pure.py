from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .matvec import GpuCcq4Session
from .quantizer import dequantize_ccq4_blocks, open_ccq4


def ccq4_path(root: Path, name: str) -> Path:
    return root / f"{name.replace('/', '_').replace(chr(92), '_').replace(':', '_')}.ccq4"


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vector_norm(x: list[float]) -> float:
    return math.sqrt(sum(v * v for v in x))


def gelu_pytorch_tanh(x: list[float]) -> list[float]:
    scale = math.sqrt(2.0 / math.pi)
    return [0.5 * v * (1.0 + math.tanh(scale * (v + 0.044715 * v * v * v))) for v in x]


def rms_norm_vector(x: list[float], weight: list[float], eps: float) -> list[float]:
    scale = 1.0 / math.sqrt(sum(v * v for v in x) / max(len(x), 1) + eps)
    return [v * scale * weight[i] for i, v in enumerate(x)]


def rms_norm_heads(x: list[float], heads: int, head_dim: int, weight: list[float], eps: float) -> list[float]:
    out: list[float] = []
    for head in range(heads):
        start = head * head_dim
        out.extend(rms_norm_vector(x[start:start + head_dim], weight, eps))
    return out


def apply_rope(vec: list[float], position: int, head_dim: int, theta: float) -> list[float]:
    out = list(vec)
    half = head_dim // 2
    for base in range(0, len(out), head_dim):
        for i in range(half):
            inv_freq = theta ** (-(i / half))
            angle = position * inv_freq
            c = math.cos(angle)
            s = math.sin(angle)
            x1 = out[base + i]
            x2 = out[base + half + i]
            out[base + i] = x1 * c - x2 * s
            out[base + half + i] = x1 * s + x2 * c
    return out


def dequantize_ccq4_row(path: str | Path, row: int) -> list[float]:
    tensor = open_ccq4(path)
    if len(tensor.shape) != 2:
        raise ValueError(f"{path} is not a matrix: shape={tensor.shape}")
    rows, cols = tensor.shape
    if row < 0 or row >= rows:
        raise ValueError(f"row {row} outside 0..{rows - 1}")
    blocks_per_row = (cols + tensor.block_size - 1) // tensor.block_size
    values = dequantize_ccq4_blocks(tensor.path, row * blocks_per_row, blocks_per_row)
    return values[:cols]


def dequantize_ccq4_rows(path: str | Path, start_row: int, row_count: int) -> list[list[float]]:
    tensor = open_ccq4(path)
    if len(tensor.shape) != 2:
        raise ValueError(f"{path} is not a matrix: shape={tensor.shape}")
    rows, cols = tensor.shape
    row_count = max(0, min(row_count, rows - start_row))
    blocks_per_row = (cols + tensor.block_size - 1) // tensor.block_size
    values = dequantize_ccq4_blocks(tensor.path, start_row * blocks_per_row, row_count * blocks_per_row)
    return [values[i * cols:(i + 1) * cols] for i in range(row_count)]


def read_ccq4_vector(path: str | Path) -> list[float]:
    tensor = open_ccq4(path)
    if len(tensor.shape) != 1:
        raise ValueError(f"{path} is not a vector: shape={tensor.shape}")
    return dequantize_ccq4_blocks(tensor.path)


def grouped_attention(
    q: list[float],
    k: list[float],
    v: list[float],
    key_cache: list[list[float]],
    value_cache: list[list[float]],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[list[float], list[float]]:
    key_cache.append(list(k))
    value_cache.append(list(v))
    repeats = q_heads // kv_heads
    context: list[float] = []
    last_scores: list[float] = []
    scale = 1.0 / math.sqrt(float(head_dim))
    for q_head in range(q_heads):
        kv_head = q_head // repeats
        q_start = q_head * head_dim
        kv_start = kv_head * head_dim
        qv = q[q_start:q_start + head_dim]
        scores = [dot(cache[kv_start:kv_start + head_dim], qv) * scale for cache in key_cache]
        max_score = max(scores)
        exps = [math.exp(score - max_score) for score in scores]
        denom = sum(exps) or 1.0
        probs = [value / denom for value in exps]
        head = [0.0] * head_dim
        for prob, cache in zip(probs, value_cache):
            vals = cache[kv_start:kv_start + head_dim]
            for i, value in enumerate(vals):
                head[i] += prob * value
        context.extend(head)
        last_scores.append(scores[-1])
    return context, last_scores


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
        self.key_cache: list[list[list[float]]] = [[] for _ in range(self.config.num_layers)]
        self.value_cache: list[list[list[float]]] = [[] for _ in range(self.config.num_layers)]
        self.norm_cache: dict[str, list[float]] = {}

    def tensor_path(self, name: str) -> Path:
        path = ccq4_path(self.ccq4_dir, name)
        if not path.exists():
            raise FileNotFoundError(f"missing CCQ4 tensor: {path}")
        return path

    def norm_weight(self, name: str) -> list[float]:
        if name not in self.norm_cache:
            self.norm_cache[name] = read_ccq4_vector(self.tensor_path(name))
        return self.norm_cache[name]

    def embedding(self, token_id: int) -> list[float]:
        return dequantize_ccq4_row(self.tensor_path("model.language_model.embed_tokens.weight"), token_id)

    def gpu_matvec(self, session: GpuCcq4Session, name: str, x: list[float], resident: bool = True) -> list[float]:
        path = self.tensor_path(name)
        return session.matvec_resident(path, x) if resident else session.matvec(path, x)

    def layer(self, session: GpuCcq4Session, x: list[float], layer_index: int, position: int, run_mlp: bool, resident: bool) -> dict:
        cfg = self.config
        prefix = f"model.language_model.layers.{layer_index}"
        residual = x
        h = rms_norm_vector(x, self.norm_weight(f"{prefix}.input_layernorm.weight"), cfg.rms_norm_eps)
        q = self.gpu_matvec(session, f"{prefix}.self_attn.q_proj.weight", h, resident)
        k = self.gpu_matvec(session, f"{prefix}.self_attn.k_proj.weight", h, resident)
        v = self.gpu_matvec(session, f"{prefix}.self_attn.v_proj.weight", h, resident)
        q = rms_norm_heads(q, cfg.q_heads, cfg.head_dim, self.norm_weight(f"{prefix}.self_attn.q_norm.weight"), cfg.rms_norm_eps)
        k = rms_norm_heads(k, cfg.kv_heads, cfg.head_dim, self.norm_weight(f"{prefix}.self_attn.k_norm.weight"), cfg.rms_norm_eps)
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
        attn_out = self.gpu_matvec(session, f"{prefix}.self_attn.o_proj.weight", context, resident)
        x = [a + b for a, b in zip(residual, attn_out)]
        mlp_summary = None
        if run_mlp:
            residual = x
            h = rms_norm_vector(x, self.norm_weight(f"{prefix}.pre_feedforward_layernorm.weight"), cfg.rms_norm_eps)
            gate = self.gpu_matvec(session, f"{prefix}.mlp.gate_proj.weight", h, resident)
            up = self.gpu_matvec(session, f"{prefix}.mlp.up_proj.weight", h, resident)
            gelu = gelu_pytorch_tanh(gate)
            ff = [a * b for a, b in zip(gelu, up)]
            down = self.gpu_matvec(session, f"{prefix}.mlp.down_proj.weight", ff, resident)
            x = [a + b for a, b in zip(residual, down)]
            mlp_summary = {"intermediate": len(ff), "down_norm": vector_norm(down)}
        return {
            "hidden": x,
            "attention": {
                "q_norm": vector_norm(q),
                "k_norm": vector_norm(k),
                "v_norm": vector_norm(v),
                "out_norm": vector_norm(attn_out),
                "last_scores": scores,
                "cache_len": len(self.key_cache[layer_index]),
            },
            "mlp": mlp_summary,
        }

    def logits_topk(self, hidden: list[float], top_k: int, vocab_limit: int | None = None, chunk_rows: int = 256) -> list[dict[str, float | int]]:
        final = rms_norm_vector(hidden, self.norm_weight("model.language_model.norm.weight"), self.config.rms_norm_eps)
        embed_path = self.tensor_path("model.language_model.embed_tokens.weight")
        tensor = open_ccq4(embed_path)
        rows, _ = tensor.shape
        limit = min(vocab_limit or rows, rows)
        best: list[tuple[float, int]] = []
        for start in range(0, limit, chunk_rows):
            matrix = dequantize_ccq4_rows(embed_path, start, min(chunk_rows, limit - start))
            for offset, row in enumerate(matrix):
                logit = dot(row, final)
                best.append((logit, start + offset))
            best = sorted(best, reverse=True)[:top_k]
        return [{"token_id": token_id, "logit": logit} for logit, token_id in best]
