from __future__ import annotations

import argparse
import json
from pathlib import Path

from .planner import GemmaSubstratePlanner


def parse_emotion(value: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("emotion must be precision,novelty,dream,risk")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan a Gemma deep-substrate runtime from a manifest.")
    parser.add_argument("--manifest", default="build/gemma_substrate_manifest.json")
    parser.add_argument("--dll", default="build/CC_OpenCl.dll")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", choices=["host", "gpu"], default="gpu")
    parser.add_argument("--emotion", type=parse_emotion, default=(0.55, 0.45, 0.15, 0.10))
    parser.add_argument("--relevance-threshold", type=float, default=0.20)
    parser.add_argument("--exploration-bias", type=float, default=0.20)
    parser.add_argument("--out", default="build/gemma_runtime_plan.json")
    args = parser.parse_args()

    planner = GemmaSubstratePlanner(args.manifest)
    if args.mode == "gpu":
        plan = planner.gpu_plan(
            dll=args.dll,
            gpu=args.gpu,
            emotion_state=args.emotion,
            relevance_threshold=args.relevance_threshold,
            exploration_bias=args.exploration_bias,
        )
    else:
        plan = planner.host_plan(relevance_threshold=args.relevance_threshold)

    data = plan.to_json_dict()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    route_histogram: dict[str, int] = {}
    for route in plan.route_names:
        route_histogram[route] = route_histogram.get(route, 0) + 1

    print(json.dumps({
        "wrote": str(out),
        "model": plan.model,
        "layer_count": plan.layer_count,
        "route_histogram": route_histogram,
        "active_layers": len(plan.active_layers),
        "budget": plan.budget,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
