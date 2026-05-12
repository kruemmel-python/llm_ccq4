from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import random
import struct
from pathlib import Path
from typing import Iterable

from .quantizer import Ccq4Tensor, dequantize_ccq4_blocks, open_ccq4
from subqg_tests.cc_driver import DriverSession, f32_array, ptr, require_buffer, require_ok


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def deterministic_vector(size: int, seed: int = 1234) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(size)]


def read_f32_vector(path: str | Path) -> list[float]:
    data = Path(path).read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} length is not divisible by 4")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def write_f32_vector(path: str | Path, values: Iterable[float]) -> None:
    vals = list(values)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(struct.pack(f"<{len(vals)}f", *vals))


def ccq4_matvec(path: str | Path, x: list[float]) -> list[float]:
    tensor = open_ccq4(path)
    if len(tensor.shape) != 2:
        raise ValueError(f"{path} is not a rank-2 matrix: shape={tensor.shape}")
    rows, cols = tensor.shape
    if len(x) != cols:
        raise ValueError(f"input length {len(x)} does not match matrix cols {cols}")
    blocks_per_row = (cols + tensor.block_size - 1) // tensor.block_size
    y = [0.0] * rows
    for row in range(rows):
        row_sum = 0.0
        for block_in_row in range(blocks_per_row):
            block_index = row * blocks_per_row + block_in_row
            block_values = dequantize_ccq4_blocks(tensor.path, start_block=block_index, block_count=1)
            col_start = block_in_row * tensor.block_size
            x_block = x[col_start:col_start + len(block_values)]
            row_sum += dot(block_values, x_block)
        y[row] = row_sum
    return y


def ccq4_matvec_summary(tensor: Ccq4Tensor, y: list[float]) -> dict[str, float | int | list[int]]:
    max_abs = max((abs(v) for v in y), default=0.0)
    mean_abs = sum(abs(v) for v in y) / max(len(y), 1)
    l2 = math.sqrt(sum(v * v for v in y))
    return {
        "shape": list(tensor.shape),
        "rows": tensor.shape[0] if tensor.shape else 0,
        "cols": tensor.shape[1] if len(tensor.shape) > 1 else 0,
        "output_count": len(y),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "l2": l2,
    }


def read_ccq4_payload(path: str | Path) -> tuple[Ccq4Tensor, bytes, bytes]:
    tensor = open_ccq4(path)
    scale_bytes = int(tensor.header["scale_bytes"])
    data_bytes = int(tensor.header["data_bytes"])
    with tensor.path.open("rb") as f:
        f.seek(tensor.scales_offset)
        scales = f.read(scale_bytes)
        f.seek(tensor.data_offset)
        packed = f.read(data_bytes)
    if len(scales) != scale_bytes or len(packed) != data_bytes:
        raise ValueError(f"{path} has truncated CCQ4 payload")
    return tensor, scales, packed


class GpuCcq4Session:
    def __init__(self, dll: str | Path, gpu: int = 0):
        self.dll = str(dll)
        self.gpu = gpu
        self._session: DriverSession | None = None
        self.lib = None
        self._resident: dict[str, ResidentCcq4Matrix] = {}

    def __enter__(self) -> "GpuCcq4Session":
        self._session = DriverSession(self.dll, self.gpu)
        self.lib = self._session.__enter__()
        if not hasattr(self.lib, "execute_ccq4_matvec_gpu"):
            raise RuntimeError("execute_ccq4_matvec_gpu export missing")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for matrix in list(self._resident.values()):
            matrix.release()
        self._resident.clear()
        if self._session is not None:
            self._session.__exit__(exc_type, exc, tb)
        self._session = None
        self.lib = None

    def matvec(self, path: str | Path, x: list[float]) -> list[float]:
        if self.lib is None:
            raise RuntimeError("GpuCcq4Session is not initialized")

        tensor, scales, packed = read_ccq4_payload(path)
        if len(tensor.shape) != 2:
            raise ValueError(f"{path} is not a rank-2 matrix: shape={tensor.shape}")
        rows, cols = tensor.shape
        if len(x) != cols:
            raise ValueError(f"input length {len(x)} does not match matrix cols {cols}")

        packed_arr = (C.c_ubyte * len(packed)).from_buffer_copy(packed)
        scales_arr = (C.c_float * (len(scales) // 4)).from_buffer_copy(scales)
        x_arr = f32_array(x)
        y_arr = (C.c_float * rows)()

        lib = self.lib
        buf_packed = require_buffer(lib.allocate_gpu_memory(self.gpu, C.sizeof(packed_arr)), "alloc packed")
        buf_scales = require_buffer(lib.allocate_gpu_memory(self.gpu, C.sizeof(scales_arr)), "alloc scales")
        buf_x = require_buffer(lib.allocate_gpu_memory(self.gpu, C.sizeof(x_arr)), "alloc x")
        buf_y = require_buffer(lib.allocate_gpu_memory(self.gpu, C.sizeof(y_arr)), "alloc y")
        try:
            require_ok(lib, lib.write_host_to_gpu_blocking(self.gpu, buf_packed, 0, C.sizeof(packed_arr), ptr(packed_arr)), "write packed")
            require_ok(lib, lib.write_host_to_gpu_blocking(self.gpu, buf_scales, 0, C.sizeof(scales_arr), ptr(scales_arr)), "write scales")
            require_ok(lib, lib.write_host_to_gpu_blocking(self.gpu, buf_x, 0, C.sizeof(x_arr), ptr(x_arr)), "write x")
            require_ok(
                lib,
                lib.execute_ccq4_matvec_gpu(self.gpu, buf_packed, buf_scales, buf_x, buf_y, rows, cols, tensor.block_size),
                "execute_ccq4_matvec_gpu",
            )
            require_ok(lib, lib.finish_gpu(self.gpu), "finish_gpu")
            require_ok(lib, lib.read_gpu_to_host_blocking(self.gpu, buf_y, 0, C.sizeof(y_arr), ptr(y_arr)), "read y")
        finally:
            lib.free_gpu_memory(self.gpu, buf_packed)
            lib.free_gpu_memory(self.gpu, buf_scales)
            lib.free_gpu_memory(self.gpu, buf_x)
            lib.free_gpu_memory(self.gpu, buf_y)

        return [float(y_arr[i]) for i in range(rows)]

    def resident_matrix(self, path: str | Path) -> "ResidentCcq4Matrix":
        key = str(Path(path).resolve())
        matrix = self._resident.get(key)
        if matrix is None:
            matrix = ResidentCcq4Matrix(self, path)
            self._resident[key] = matrix
        return matrix

    def matvec_resident(self, path: str | Path, x: list[float]) -> list[float]:
        return self.resident_matrix(path).matvec(x)

    @property
    def resident_count(self) -> int:
        return len(self._resident)


class ResidentCcq4Matrix:
    def __init__(self, session: GpuCcq4Session, path: str | Path):
        if session.lib is None:
            raise RuntimeError("GpuCcq4Session is not initialized")
        self.session = session
        self.path = Path(path)
        self.tensor, scales, packed = read_ccq4_payload(path)
        if len(self.tensor.shape) != 2:
            raise ValueError(f"{path} is not a rank-2 matrix: shape={self.tensor.shape}")
        self.rows, self.cols = self.tensor.shape
        self.lib = session.lib
        self.gpu = session.gpu
        self.handle = 0
        self._driver_resident = hasattr(self.lib, "cc_register_persistent_ccq4_weight") and hasattr(self.lib, "cc_execute_resident_ccq4_matvec")
        self._packed_arr = (C.c_ubyte * len(packed)).from_buffer_copy(packed)
        self._scales_arr = (C.c_float * (len(scales) // 4)).from_buffer_copy(scales)
        self.buf_packed = None
        self.buf_scales = None
        self._released = False
        if self._driver_resident:
            handle = C.c_int(0)
            require_ok(
                self.lib,
                self.lib.cc_register_persistent_ccq4_weight(
                    self.gpu,
                    ptr(self._packed_arr), C.sizeof(self._packed_arr),
                    ptr(self._scales_arr), C.sizeof(self._scales_arr),
                    self.rows, self.cols, self.tensor.block_size,
                    C.byref(handle),
                ),
                f"cc_register_persistent_ccq4_weight {self.path}",
            )
            self.handle = int(handle.value)
            if self.handle <= 0:
                raise RuntimeError(f"cc_register_persistent_ccq4_weight returned invalid handle {self.handle}")
        else:
            self.buf_packed = require_buffer(
                self.lib.allocate_gpu_memory(self.gpu, C.sizeof(self._packed_arr)),
                f"alloc resident packed {self.path}",
            )
            self.buf_scales = require_buffer(
                self.lib.allocate_gpu_memory(self.gpu, C.sizeof(self._scales_arr)),
                f"alloc resident scales {self.path}",
            )
            try:
                require_ok(
                    self.lib,
                    self.lib.write_host_to_gpu_blocking(self.gpu, self.buf_packed, 0, C.sizeof(self._packed_arr), ptr(self._packed_arr)),
                    f"write resident packed {self.path}",
                )
                require_ok(
                    self.lib,
                    self.lib.write_host_to_gpu_blocking(self.gpu, self.buf_scales, 0, C.sizeof(self._scales_arr), ptr(self._scales_arr)),
                    f"write resident scales {self.path}",
                )
            except Exception:
                self.release()
                raise

    def release(self) -> None:
        if self._released:
            return
        if self._driver_resident and self.handle > 0:
            try:
                self.lib.cc_release_persistent_weight(self.handle)
            finally:
                self.handle = 0
                self._released = True
            return
        try:
            if self.buf_packed is not None:
                self.lib.free_gpu_memory(self.gpu, self.buf_packed)
        finally:
            if self.buf_scales is not None:
                self.lib.free_gpu_memory(self.gpu, self.buf_scales)
            self._released = True

    def matvec(self, x: list[float]) -> list[float]:
        if self._released:
            raise RuntimeError(f"resident matrix already released: {self.path}")
        if len(x) != self.cols:
            raise ValueError(f"input length {len(x)} does not match matrix cols {self.cols}")
        x_arr = f32_array(x)
        y_arr = (C.c_float * self.rows)()
        buf_x = require_buffer(self.lib.allocate_gpu_memory(self.gpu, C.sizeof(x_arr)), "alloc x")
        buf_y = require_buffer(self.lib.allocate_gpu_memory(self.gpu, C.sizeof(y_arr)), "alloc y")
        try:
            require_ok(self.lib, self.lib.write_host_to_gpu_blocking(self.gpu, buf_x, 0, C.sizeof(x_arr), ptr(x_arr)), "write x")
            if self._driver_resident:
                require_ok(
                    self.lib,
                    self.lib.cc_execute_resident_ccq4_matvec(self.gpu, self.handle, buf_x, buf_y),
                    "cc_execute_resident_ccq4_matvec",
                )
            else:
                require_ok(
                    self.lib,
                    self.lib.execute_ccq4_matvec_gpu(
                        self.gpu,
                        self.buf_packed,
                        self.buf_scales,
                        buf_x,
                        buf_y,
                        self.rows,
                        self.cols,
                        self.tensor.block_size,
                    ),
                    "execute_ccq4_matvec_gpu",
                )
            require_ok(self.lib, self.lib.finish_gpu(self.gpu), "finish_gpu")
            require_ok(self.lib, self.lib.read_gpu_to_host_blocking(self.gpu, buf_y, 0, C.sizeof(y_arr), ptr(y_arr)), "read y")
        finally:
            self.lib.free_gpu_memory(self.gpu, buf_x)
            self.lib.free_gpu_memory(self.gpu, buf_y)
        return [float(y_arr[i]) for i in range(self.rows)]


def ccq4_matvec_gpu(path: str | Path, x: list[float], dll: str | Path, gpu: int = 0) -> list[float]:
    with GpuCcq4Session(dll, gpu) as session:
        return session.matvec(path, x)


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU reference matvec for CCQ4 tensors.")
    parser.add_argument("--ccq4", required=True)
    parser.add_argument("--input-f32", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-f32", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--gpu-dll", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--compare-gpu", action="store_true")
    args = parser.parse_args()

    try:
        tensor = open_ccq4(args.ccq4)
        if len(tensor.shape) != 2:
            raise ValueError(f"CCQ4 tensor must be rank-2 for matvec, got shape={tensor.shape}")
        cols = tensor.shape[1]
        x = read_f32_vector(args.input_f32) if args.input_f32 else deterministic_vector(cols, args.seed)
        y = ccq4_matvec(args.ccq4, x)
        gpu_summary = None
        if args.compare_gpu:
            if not args.gpu_dll:
                raise ValueError("--gpu-dll is required with --compare-gpu")
            y_gpu = ccq4_matvec_gpu(args.ccq4, x, args.gpu_dll, args.gpu)
            max_abs_error = max(abs(a - b) for a, b in zip(y_gpu, y)) if y else 0.0
            gpu_summary = {
                "gpu_output_count": len(y_gpu),
                "gpu_max_abs_error_vs_cpu": max_abs_error,
                "gpu_first_values": y_gpu[:8],
            }
        if args.output_f32:
            write_f32_vector(args.output_f32, y)
        summary = ccq4_matvec_summary(tensor, y)
        summary["ccq4"] = str(Path(args.ccq4))
        if gpu_summary:
            summary["gpu"] = gpu_summary
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"gemma_runtime.matvec: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
