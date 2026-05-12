__all__ = [
    "GemmaSubstratePlanner",
    "RuntimePlan",
    "GemmaWeightIndex",
    "SafetensorsShard",
    "TensorRef",
]


def __getattr__(name):
    if name in {"GemmaSubstratePlanner", "RuntimePlan"}:
        from .planner import GemmaSubstratePlanner, RuntimePlan

        return {"GemmaSubstratePlanner": GemmaSubstratePlanner, "RuntimePlan": RuntimePlan}[name]
    if name in {"GemmaWeightIndex", "SafetensorsShard", "TensorRef"}:
        from .weights import GemmaWeightIndex, SafetensorsShard, TensorRef

        return {
            "GemmaWeightIndex": GemmaWeightIndex,
            "SafetensorsShard": SafetensorsShard,
            "TensorRef": TensorRef,
        }[name]
    raise AttributeError(name)
