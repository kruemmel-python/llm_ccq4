from __future__ import annotations

import ctypes as C
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from subqg_tests.cc_driver import DriverSession, f32_array, i32_array, ptr, require_buffer, require_ok


ROUTE_NAMES = {
    0: "dense",
    1: "mycel",
    2: "tensor_bond",
    3: "reservoir",
    4: "langevin",
    5: "subqg_quantum",
}

ROLE_TO_DEFAULT_ROUTE = {
    0: 0,
    1: 5,
    2: 3,
    3: 0,
    4: 1,
}


def role_from_counts(role_counts: dict[str, int], fallback: int) -> int:
    attention = int(role_counts.get("attention", 0))
    mlp = int(role_counts.get("mlp", 0))
    router = int(role_counts.get("sparse_router", 0))
    norm = int(role_counts.get("norm_residual", 0))
    dense = int(role_counts.get("dense", 0))
    total = max(attention + mlp + router + norm + dense, 1)
    if router >= max(4, attention + mlp):
        return 4
    if attention >= max(4, mlp * 2):
        return 1
    if mlp >= 2:
        return 2
    if norm and norm >= total * 0.50:
        return 3
    if attention:
        return 1
    return fallback


@dataclass(frozen=True)
class RuntimePlan:
    model: str
    layer_count: int
    route_count: int
    selected_route: list[int]
    route_names: list[str]
    dispatch_params: list[list[float]]
    active_layers: list[int]
    budget: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "layer_count": self.layer_count,
            "route_count": self.route_count,
            "selected_route": self.selected_route,
            "route_names": self.route_names,
            "dispatch_params": self.dispatch_params,
            "active_layers": self.active_layers,
            "budget": self.budget,
        }


class GemmaSubstratePlanner:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        dispatcher = self.manifest.get("dispatcher", {})
        self.layer_role = [int(v) for v in dispatcher.get("layer_role", [])]
        self.layer_priority = [float(v) for v in dispatcher.get("layer_priority", [])]
        layers = self.manifest.get("layers", [])
        if isinstance(layers, list) and len(layers) == len(self.layer_role):
            refined_roles = []
            for idx, role in enumerate(self.layer_role):
                layer = layers[idx]
                counts = layer.get("role_counts", {}) if isinstance(layer, dict) else {}
                refined_roles.append(role_from_counts(counts, role) if isinstance(counts, dict) else role)
            self.layer_role = refined_roles
        self.route_count = int(dispatcher.get("route_count", 6))
        if len(self.layer_role) != len(self.layer_priority):
            raise ValueError("manifest dispatcher layer_role/layer_priority length mismatch")
        if self.route_count < 6:
            raise ValueError("dispatcher route_count must be at least 6")

    @property
    def layer_count(self) -> int:
        return len(self.layer_role)

    def host_plan(
        self,
        relevance_threshold: float = 0.20,
    ) -> RuntimePlan:
        selected = []
        active = []
        params = []
        for idx, role in enumerate(self.layer_role):
            route = ROLE_TO_DEFAULT_ROUTE.get(role, 0)
            selected.append(route)
            relevance = max(0.0, min(1.0, self.layer_priority[idx]))
            params.append([0.50, 0.25, 0.10, relevance])
            if relevance >= relevance_threshold:
                active.append(idx)
        return RuntimePlan(
            model=str(self.manifest.get("model", "unknown")),
            layer_count=self.layer_count,
            route_count=self.route_count,
            selected_route=selected,
            route_names=[ROUTE_NAMES.get(route, f"route_{route}") for route in selected],
            dispatch_params=params,
            active_layers=active,
            budget=dict(self.manifest.get("budget", {})),
        )

    def gpu_plan(
        self,
        dll: str | Path,
        gpu: int = 0,
        emotion_state: tuple[float, float, float, float] = (0.55, 0.45, 0.15, 0.10),
        nutrient_state: list[float] | None = None,
        relevance_threshold: float = 0.20,
        exploration_bias: float = 0.20,
    ) -> RuntimePlan:
        if nutrient_state is None:
            nutrient_state = [1.0] * self.layer_count
        if len(nutrient_state) != self.layer_count:
            raise ValueError("nutrient_state length must match layer_count")

        roles = i32_array(self.layer_role)
        priorities = f32_array(self.layer_priority)
        emotions = f32_array(emotion_state)
        nutrients = f32_array(nutrient_state)

        scores = (C.c_float * (self.layer_count * self.route_count))()
        selected = (C.c_int * self.layer_count)()
        params = (C.c_float * (self.layer_count * 4))()
        active_layers = (C.c_int * self.layer_count)()
        active_count = (C.c_uint * 1)()

        with DriverSession(str(dll), gpu) as lib:
            buffers = [
                require_buffer(lib.allocate_gpu_memory(gpu, C.sizeof(roles)), "alloc roles"),
                require_buffer(lib.allocate_gpu_memory(gpu, C.sizeof(priorities)), "alloc priorities"),
                require_buffer(lib.allocate_gpu_memory(gpu, C.sizeof(emotions)), "alloc emotions"),
                require_buffer(lib.allocate_gpu_memory(gpu, C.sizeof(nutrients)), "alloc nutrients"),
                require_buffer(lib.allocate_gpu_memory(gpu, C.sizeof(scores)), "alloc scores"),
                require_buffer(lib.allocate_gpu_memory(gpu, C.sizeof(selected)), "alloc selected"),
                require_buffer(lib.allocate_gpu_memory(gpu, C.sizeof(params)), "alloc params"),
                require_buffer(lib.allocate_gpu_memory(gpu, C.sizeof(active_layers)), "alloc active_layers"),
                require_buffer(lib.allocate_gpu_memory(gpu, C.sizeof(active_count)), "alloc active_count"),
            ]
            try:
                buf_role, buf_priority, buf_emotion, buf_nutrient, buf_scores, buf_selected, buf_params, buf_active_layers, buf_active_count = buffers
                require_ok(lib, lib.write_host_to_gpu_blocking(gpu, buf_role, 0, C.sizeof(roles), ptr(roles)), "write roles")
                require_ok(lib, lib.write_host_to_gpu_blocking(gpu, buf_priority, 0, C.sizeof(priorities), ptr(priorities)), "write priorities")
                require_ok(lib, lib.write_host_to_gpu_blocking(gpu, buf_emotion, 0, C.sizeof(emotions), ptr(emotions)), "write emotions")
                require_ok(lib, lib.write_host_to_gpu_blocking(gpu, buf_nutrient, 0, C.sizeof(nutrients), ptr(nutrients)), "write nutrients")

                ok = lib.execute_deep_substrate_dispatch_gpu(
                    gpu,
                    buf_role,
                    buf_priority,
                    buf_emotion,
                    buf_nutrient,
                    buf_scores,
                    buf_selected,
                    buf_params,
                    buf_active_layers,
                    buf_active_count,
                    self.layer_count,
                    self.route_count,
                    C.c_float(relevance_threshold),
                    C.c_float(exploration_bias),
                )
                require_ok(lib, ok, "execute_deep_substrate_dispatch_gpu")
                require_ok(lib, lib.finish_gpu(gpu), "finish_gpu")
                require_ok(lib, lib.read_gpu_to_host_blocking(gpu, buf_selected, 0, C.sizeof(selected), ptr(selected)), "read selected")
                require_ok(lib, lib.read_gpu_to_host_blocking(gpu, buf_params, 0, C.sizeof(params), ptr(params)), "read params")
                require_ok(lib, lib.read_gpu_to_host_blocking(gpu, buf_active_layers, 0, C.sizeof(active_layers), ptr(active_layers)), "read active layers")
                require_ok(lib, lib.read_gpu_to_host_blocking(gpu, buf_active_count, 0, C.sizeof(active_count), ptr(active_count)), "read active count")
            finally:
                for buffer in buffers:
                    lib.free_gpu_memory(gpu, buffer)

        selected_list = [int(selected[i]) for i in range(self.layer_count)]
        active_n = min(int(active_count[0]), self.layer_count)
        return RuntimePlan(
            model=str(self.manifest.get("model", "unknown")),
            layer_count=self.layer_count,
            route_count=self.route_count,
            selected_route=selected_list,
            route_names=[ROUTE_NAMES.get(route, f"route_{route}") for route in selected_list],
            dispatch_params=[
                [float(params[i * 4 + j]) for j in range(4)]
                for i in range(self.layer_count)
            ],
            active_layers=[int(active_layers[i]) for i in range(active_n)],
            budget=dict(self.manifest.get("budget", {})),
        )
