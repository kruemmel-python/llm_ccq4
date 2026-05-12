from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any


DEFAULT_HF_MODEL = "google/gemma-3n-E4B"
ROUTE_COUNT = 6

ROLE_DENSE = 0
ROLE_ATTENTION = 1
ROLE_MLP = 2
ROLE_NORM = 3
ROLE_ROUTER = 4


ROLE_NAMES = {
    ROLE_DENSE: "dense",
    ROLE_ATTENTION: "attention",
    ROLE_MLP: "mlp",
    ROLE_NORM: "norm_residual",
    ROLE_ROUTER: "sparse_router",
}


ROLE_PATTERNS = [
    (ROLE_ATTENTION, ("self_attn", ".attn", "attention", ".q_proj", ".k_proj", ".v_proj", ".o_proj")),
    (ROLE_MLP, ("mlp", "feed_forward", "ffn", ".gate_proj", ".up_proj", ".down_proj")),
    (ROLE_NORM, ("norm", "layernorm", "rms_norm", "input_layernorm", "post_attention_layernorm")),
    (ROLE_ROUTER, ("router", "expert", "moe", "mixture")),
]


LAYER_RE = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.(\d+)(?:\.|$)")


def mib(num_bytes: float) -> float:
    return num_bytes / (1024.0 * 1024.0)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def hf_resolve_url(model_id: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"


def direct_hf_download(model_id: str, filename: str, revision: str, token: str | None, out_dir: pathlib.Path) -> pathlib.Path:
    url = hf_resolve_url(model_id, filename, revision)
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code in (401, 403):
            hint = (
                " Token rejected or missing access. Revoke any exposed token, create a fresh read token, "
                "and accept the model license on the Hugging Face model page."
            )
        raise RuntimeError(f"HTTP {exc.code} while downloading {filename} from {model_id}.{hint}") from exc
    path = out_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def try_hf_download(model_id: str, cache_dir: pathlib.Path | None, revision: str, token: str | None) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir = cache_dir if cache_dir else pathlib.Path(tempfile.gettempdir()) / "cc_opencl_hf_metadata" / model_id.replace("/", "__")
    try:
        config = direct_hf_download(model_id, "config.json", revision, token, out_dir)
        index = direct_hf_download(model_id, "model.safetensors.index.json", revision, token, out_dir)
        return config, index
    except Exception as direct_exc:
        direct_error = direct_exc

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Direct HTTPS metadata download failed and huggingface_hub is not installed. "
            f"Direct error: {direct_error}"
        ) from exc

    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    if token:
        kwargs["token"] = token
    if revision:
        kwargs["revision"] = revision
    try:
        config = pathlib.Path(hf_hub_download(model_id, "config.json", **kwargs))
        index = pathlib.Path(hf_hub_download(model_id, "model.safetensors.index.json", **kwargs))
        return config, index
    except TypeError:
        # Very old huggingface_hub versions do not accept token/revision kwargs.
        # Avoid retrying gated repos without auth because that hides the useful
        # direct HTTPS diagnostic behind a raw Hub 401.
        legacy_kwargs: dict[str, Any] = {}
        if token:
            legacy_kwargs["use_auth_token"] = token
        try:
            config = pathlib.Path(hf_hub_download(model_id, "config.json", **legacy_kwargs))
            index = pathlib.Path(hf_hub_download(model_id, "model.safetensors.index.json", **legacy_kwargs))
            return config, index
        except Exception as legacy_exc:
            raise RuntimeError(
                f"Direct HTTPS and legacy huggingface_hub downloads both failed. Direct error: {direct_error}"
            ) from legacy_exc
    except Exception as hub_exc:
        raise RuntimeError(f"Direct HTTPS and huggingface_hub downloads both failed. Direct error: {direct_error}") from hub_exc


def resolve_inputs(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    if args.config or args.index:
        if not args.config or not args.index:
            raise SystemExit("--config and --index must be provided together")
        return pathlib.Path(args.config), pathlib.Path(args.index)

    if args.model_dir:
        model_dir = pathlib.Path(args.model_dir)
        return model_dir / "config.json", model_dir / "model.safetensors.index.json"

    cache_dir = pathlib.Path(args.cache_dir) if args.cache_dir else None
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return try_hf_download(args.hf_model, cache_dir, args.revision, token)


def nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def infer_text_config(config: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        nested_get(config, ("text_config",)),
        nested_get(config, ("language_config",)),
        nested_get(config, ("llm_config",)),
        nested_get(config, ("model_config",)),
        config,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and (
            "num_hidden_layers" in candidate or "hidden_size" in candidate or "num_attention_heads" in candidate
        ):
            return candidate
    return config


def infer_weight_bytes(index: dict[str, Any]) -> int:
    metadata = index.get("metadata")
    if isinstance(metadata, dict):
        total = metadata.get("total_size")
        if isinstance(total, int):
            return total
        if isinstance(total, str) and total.isdigit():
            return int(total)
    return 0


def shard_sizes_from_weight_map(index: dict[str, Any]) -> dict[str, int]:
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        return {}
    shards: dict[str, int] = {}
    for shard in weight_map.values():
        if isinstance(shard, str):
            shards.setdefault(shard, 0)
    return shards


def classify_key(key: str) -> int:
    lower = key.lower()
    for role, needles in ROLE_PATTERNS:
        if any(needle in lower for needle in needles):
            return role
    return ROLE_DENSE


def layer_id_from_key(key: str) -> int | None:
    match = LAYER_RE.search(key)
    if not match:
        return None
    return int(match.group(1))


def summarize_layers(index: dict[str, Any], configured_layers: int | None) -> list[dict[str, Any]]:
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json missing object field 'weight_map'")

    per_layer_roles: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    per_layer_keys: dict[int, int] = defaultdict(int)

    for key in weight_map.keys():
        if not isinstance(key, str):
            continue
        layer_id = layer_id_from_key(key)
        if layer_id is None:
            continue
        role = classify_key(key)
        per_layer_roles[layer_id][role] += 1
        per_layer_keys[layer_id] += 1

    max_seen = max(per_layer_keys.keys(), default=-1)
    layer_count = configured_layers if configured_layers and configured_layers > 0 else max_seen + 1

    layers: list[dict[str, Any]] = []
    for layer in range(layer_count):
        role_counts = dict(per_layer_roles.get(layer, {}))
        attention = role_counts.get(ROLE_ATTENTION, 0)
        mlp = role_counts.get(ROLE_MLP, 0)
        router = role_counts.get(ROLE_ROUTER, 0)
        norm = role_counts.get(ROLE_NORM, 0)
        total = max(per_layer_keys.get(layer, 0), 1)

        if router >= max(4, attention + mlp):
            role = ROLE_ROUTER
        elif attention >= max(4, mlp * 2):
            role = ROLE_ATTENTION
        elif mlp >= 2:
            role = ROLE_MLP
        elif norm and norm >= total * 0.50:
            role = ROLE_NORM
        elif attention:
            role = ROLE_ATTENTION
        elif role_counts:
            role = max(role_counts.items(), key=lambda item: item[1])[0]
        else:
            role = ROLE_DENSE

        priority = 0.55 + min(0.45, math.log2(total + 1.0) / 12.0)
        if attention and mlp:
            priority += 0.05
        if router:
            priority += 0.05
        if norm and not (attention or mlp):
            priority -= 0.15
        priority = max(0.10, min(1.0, priority))
        layers.append(
            {
                "layer": layer,
                "role": role,
                "role_name": ROLE_NAMES[role],
                "priority": round(priority, 4),
                "weight_key_count": per_layer_keys.get(layer, 0),
                "role_counts": {ROLE_NAMES[k]: v for k, v in sorted(role_counts.items())},
            }
        )
    return layers


def estimate_quantized_budget(weight_bytes: int, safe_vram_gib: float, substrate_mib: float) -> dict[str, Any]:
    safe_bytes = int(safe_vram_gib * 1024 * 1024 * 1024 * 0.65)
    substrate_bytes = int(substrate_mib * 1024 * 1024)
    available_for_weights = max(0, safe_bytes - substrate_bytes)
    if weight_bytes <= 0:
        return {
            "raw_weight_mib": None,
            "safe_vram_mib": round(mib(safe_bytes), 2),
            "substrate_reserved_mib": round(substrate_mib, 2),
            "available_weight_mib": round(mib(available_for_weights), 2),
            "note": "No total_size metadata found; inspect shard files after download.",
        }

    quantized = {
        "fp16_mib": mib(weight_bytes),
        "int8_mib": mib(weight_bytes * 0.50),
        "int4_mib": mib(weight_bytes * 0.25),
        "int3_mib": mib(weight_bytes * 0.1875),
        "int2_mib": mib(weight_bytes * 0.125),
    }
    feasible = {
        name: value * 1024 * 1024 <= available_for_weights
        for name, value in quantized.items()
    }
    return {
        "raw_weight_mib": round(mib(weight_bytes), 2),
        "safe_vram_mib": round(mib(safe_bytes), 2),
        "substrate_reserved_mib": round(substrate_mib, 2),
        "available_weight_mib": round(mib(available_for_weights), 2),
        "quantized_estimates_mib": {k: round(v, 2) for k, v in quantized.items()},
        "fits_in_available_weight_budget": feasible,
    }


def build_manifest(config: dict[str, Any], index: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    text_config = infer_text_config(config)
    configured_layers = text_config.get("num_hidden_layers")
    if not isinstance(configured_layers, int):
        configured_layers = None

    layers = summarize_layers(index, configured_layers)
    role_histogram: dict[str, int] = defaultdict(int)
    for layer in layers:
        role_histogram[layer["role_name"]] += 1

    weight_bytes = infer_weight_bytes(index)
    budget = estimate_quantized_budget(weight_bytes, args.safe_vram_gib, args.substrate_mib)

    return {
        "model": args.hf_model,
        "config_model_type": config.get("model_type"),
        "text_config": {
            "num_hidden_layers": configured_layers,
            "hidden_size": text_config.get("hidden_size"),
            "intermediate_size": text_config.get("intermediate_size"),
            "num_attention_heads": text_config.get("num_attention_heads"),
            "num_key_value_heads": text_config.get("num_key_value_heads"),
            "vocab_size": text_config.get("vocab_size"),
        },
        "dispatcher": {
            "route_count": ROUTE_COUNT,
            "layer_role": [layer["role"] for layer in layers],
            "layer_priority": [layer["priority"] for layer in layers],
            "role_names": ROLE_NAMES,
        },
        "layers": layers,
        "role_histogram": dict(sorted(role_histogram.items())),
        "shards": sorted(shard_sizes_from_weight_map(index).keys()),
        "budget": budget,
    }


def write_outputs(manifest: dict[str, Any], out_path: pathlib.Path | None, c_arrays_path: pathlib.Path | None) -> None:
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(manifest, indent=2))

    if c_arrays_path:
        roles = ", ".join(str(v) for v in manifest["dispatcher"]["layer_role"])
        priorities = ", ".join(f"{v:.4f}f" for v in manifest["dispatcher"]["layer_priority"])
        layer_count = len(manifest["dispatcher"]["layer_role"])
        c_arrays_path.parent.mkdir(parents=True, exist_ok=True)
        c_arrays_path.write_text(
            "\n".join(
                [
                    "/* Generated by scripts/gemma_substrate_inspector.py */",
                    f"#define GEMMA_SUBSTRATE_LAYER_COUNT {layer_count}",
                    f"#define GEMMA_SUBSTRATE_ROUTE_COUNT {ROUTE_COUNT}",
                    f"static const int gemma_layer_role[GEMMA_SUBSTRATE_LAYER_COUNT] = {{{roles}}};",
                    f"static const float gemma_layer_priority[GEMMA_SUBSTRATE_LAYER_COUNT] = {{{priorities}}};",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Gemma safetensors metadata and emit dispatcher-ready substrate mapping."
    )
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--hf-token", default=None, help="Optional token. Prefer HF_TOKEN env var so it is not stored in shell history.")
    parser.add_argument("--model-dir", default=None, help="Directory containing config.json and model.safetensors.index.json.")
    parser.add_argument("--config", default=None, help="Explicit config.json path.")
    parser.add_argument("--index", default=None, help="Explicit model.safetensors.index.json path.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory for metadata downloads.")
    parser.add_argument("--out", default="build/gemma_substrate_manifest.json")
    parser.add_argument("--c-arrays-out", default=None)
    parser.add_argument("--safe-vram-gib", type=float, default=4.0)
    parser.add_argument("--substrate-mib", type=float, default=384.0)
    args = parser.parse_args(argv)

    try:
        config_path, index_path = resolve_inputs(args)
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found: {config_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"model.safetensors.index.json not found: {index_path}")
        config = load_json(config_path)
        index = load_json(index_path)
        manifest = build_manifest(config, index, args)
        out_path = pathlib.Path(args.out) if args.out else None
        c_arrays_path = pathlib.Path(args.c_arrays_out) if args.c_arrays_out else None
        write_outputs(manifest, out_path, c_arrays_path)
        if out_path:
            print(f"Wrote {out_path}")
        if c_arrays_path:
            print(f"Wrote {c_arrays_path}")
        return 0
    except Exception as exc:
        print(f"gemma_substrate_inspector: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
