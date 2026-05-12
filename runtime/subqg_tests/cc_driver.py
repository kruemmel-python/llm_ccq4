from __future__ import annotations

import ctypes as C
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, Optional


class DriverError(RuntimeError):
    pass


FloatArray = C.Array
IntArray = C.Array


class Float2(C.Structure):
    _fields_ = [("s", C.c_float * 2)]


class QuantumGate(C.Structure):
    _fields_ = [
        ("name", C.c_char * 8),
        ("arity", C.c_uint),
        ("control", C.c_uint),
        ("target", C.c_uint),
        ("control2", C.c_uint),
        ("params", C.c_float * 4),
        ("matrix", (Float2 * 8) * 8),
    ]


class PauliZTerm(C.Structure):
    _fields_ = [
        ("z_mask", C.c_uint64),
        ("coefficient", C.c_float),
    ]


class KernelMetricsSample(C.Structure):
    _fields_ = [
        ("name", C.c_char * 64),
        ("duration_ms", C.c_float),
        ("error", C.c_float),
        ("variance", C.c_float),
    ]


class QuantumEchoProfile(C.Structure):
    _fields_ = [
        ("single_qubit_gate_count", C.c_uint64),
        ("two_qubit_gate_count", C.c_uint64),
        ("three_qubit_gate_count", C.c_uint64),
        ("fused_single_gate_groups", C.c_uint64),
        ("total_gate_applications", C.c_uint64),
        ("estimated_global_mem_bytes", C.c_uint64),
        ("kernel_enqueue_count", C.c_uint64),
        ("host_wall_time_ms", C.c_double),
        ("used_out_of_order_queue", C.c_int),
    ]


@dataclass(frozen=True)
class DriverPaths:
    dll: pathlib.Path
    root: pathlib.Path


def _candidate_paths() -> list[pathlib.Path]:
    env = os.environ.get("CC_OPENCL_DLL")
    candidates: list[pathlib.Path] = []
    if env:
        candidates.append(pathlib.Path(env))

    here = pathlib.Path.cwd()
    for base in [here, here.parent, pathlib.Path(__file__).resolve().parents[1]]:
        candidates.extend([
            base / "CC_OpenCl.dll",
            base / "build" / "CC_OpenCl.dll",
            base / "build" / "bin" / "CC_OpenCl.dll",
            base / "bin" / "CC_OpenCl.dll",
            base / "CC_OpenCl_Enterprise-main" / "build" / "CC_OpenCl.dll",
            base / "CC_OpenCl_Enterprise-main" / "build" / "bin" / "CC_OpenCl.dll",
        ])
    # Remove duplicates while preserving order.
    seen: set[pathlib.Path] = set()
    uniq: list[pathlib.Path] = []
    for p in candidates:
        p = p.expanduser().resolve()
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def find_driver(dll: str | os.PathLike[str] | None = None) -> DriverPaths:
    if dll:
        path = pathlib.Path(dll).expanduser().resolve()
        if not path.exists():
            raise DriverError(f"DLL nicht gefunden: {path}")
        return DriverPaths(dll=path, root=path.parent)

    for p in _candidate_paths():
        if p.exists():
            return DriverPaths(dll=p, root=p.parent)
    tried = "\n".join(f"  - {p}" for p in _candidate_paths())
    raise DriverError(
        "CC_OpenCl.dll nicht gefunden. Setze CC_OPENCL_DLL oder übergib --dll.\n"
        f"Gesuchte Pfade:\n{tried}"
    )


def _add_dll_directories(paths: Iterable[pathlib.Path]) -> None:
    if os.name != "nt":
        return
    for path in paths:
        if path.exists() and path.is_dir():
            try:
                os.add_dll_directory(str(path))
            except (OSError, AttributeError):
                pass


def load_driver(dll: str | os.PathLike[str] | None = None) -> C.CDLL:
    paths = find_driver(dll)
    root = paths.dll.parent
    project_root = root
    # In der gelieferten ZIP liegt OpenCL.dll oft in CL/ neben dem Projektroot.
    possible_dirs = [
        root,
        root / "CL",
        root.parent / "CL",
        root.parent.parent / "CL",
        project_root,
    ]
    _add_dll_directories(possible_dirs)
    lib = C.CDLL(str(paths.dll))
    bind_api(lib)
    return lib


def bind_api(lib: C.CDLL) -> None:
    # Diagnostics
    lib.cc_get_last_error.argtypes = []
    lib.cc_get_last_error.restype = C.c_char_p
    lib.cc_get_version.argtypes = []
    lib.cc_get_version.restype = C.c_char_p

    # Lifecycle
    lib.initialize_gpu.argtypes = [C.c_int]
    lib.initialize_gpu.restype = C.c_int
    lib.shutdown_gpu.argtypes = [C.c_int]
    lib.shutdown_gpu.restype = None
    lib.finish_gpu.argtypes = [C.c_int]
    lib.finish_gpu.restype = C.c_int

    # Raw buffers
    lib.allocate_gpu_memory.argtypes = [C.c_int, C.c_size_t]
    lib.allocate_gpu_memory.restype = C.c_void_p
    lib.free_gpu_memory.argtypes = [C.c_int, C.c_void_p]
    lib.free_gpu_memory.restype = None
    lib.write_host_to_gpu_blocking.argtypes = [C.c_int, C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p]
    lib.write_host_to_gpu_blocking.restype = C.c_int
    lib.read_gpu_to_host_blocking.argtypes = [C.c_int, C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p]
    lib.read_gpu_to_host_blocking.restype = C.c_int

    # Basic kernels
    lib.execute_clone_on_gpu.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_size_t]
    lib.execute_clone_on_gpu.restype = C.c_int
    lib.execute_add_on_gpu.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
    lib.execute_add_on_gpu.restype = C.c_int

    # Ising
    lib.execute_ising_metropolis_step_on_gpu.argtypes = [
        C.c_int, C.c_void_p, C.c_void_p, C.c_float, C.c_float, C.c_int, C.c_int, C.c_int
    ]
    lib.execute_ising_metropolis_step_on_gpu.restype = C.c_int

    # SubQG
    lib.subqg_initialize_state_batched.argtypes = [
        C.c_int, C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_float, C.c_float
    ]
    lib.subqg_initialize_state_batched.restype = C.c_int
    lib.subqg_simulation_step_batched.argtypes = [
        C.c_int,
        C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float),
        C.c_int,
        C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float),
        C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int),
        C.POINTER(C.c_float), C.c_int
    ]
    lib.subqg_simulation_step_batched.restype = C.c_int
    lib.subqg_release_state.argtypes = [C.c_int]
    lib.subqg_release_state.restype = None
    lib.subqg_set_deterministic_mode.argtypes = [C.c_int, C.c_uint64]
    lib.subqg_set_deterministic_mode.restype = None
    lib.subqg_set_multifield_state.argtypes = [
        C.c_int, C.c_int,
        C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float),
        C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float)
    ]
    lib.subqg_set_multifield_state.restype = C.c_int
    lib.subqg_get_multifield_state.argtypes = [
        C.c_int, C.c_int,
        C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float),
        C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float)
    ]
    lib.subqg_get_multifield_state.restype = C.c_int

    # Quantum/QAOA/OTOC
    lib.execute_qaoa_gpu.argtypes = [
        C.c_int, C.c_int, C.c_int,
        C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_int,
        C.POINTER(PauliZTerm), C.c_int,
        C.POINTER(C.c_float)
    ]
    lib.execute_qaoa_gpu.restype = C.c_int
    lib.execute_quantum_echoes_otoc_gpu.argtypes = [
        C.c_int, C.c_int,
        C.POINTER(QuantumGate), C.c_int,
        C.POINTER(QuantumGate), C.POINTER(QuantumGate),
        C.c_int,
        C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float)
    ]
    lib.execute_quantum_echoes_otoc_gpu.restype = C.c_int
    lib.get_last_quantum_echo_profile.argtypes = [C.POINTER(QuantumEchoProfile)]
    lib.get_last_quantum_echo_profile.restype = C.c_int

    try:
        lib.get_last_kernel_metrics.argtypes = [C.c_int, C.POINTER(KernelMetricsSample)]
        lib.get_last_kernel_metrics.restype = C.c_int
    except AttributeError:
        pass

    try:
        lib.execute_deep_substrate_dispatch_gpu.argtypes = [
            C.c_int,
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
            C.c_int, C.c_int, C.c_float, C.c_float,
        ]
        lib.execute_deep_substrate_dispatch_gpu.restype = C.c_int
    except AttributeError:
        pass

    try:
        lib.execute_ccq4_matvec_gpu.argtypes = [
            C.c_int,
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
            C.c_int, C.c_int, C.c_int,
        ]
        lib.execute_ccq4_matvec_gpu.restype = C.c_int
    except AttributeError:
        pass

    try:
        lib.cc_register_persistent_ccq4_weight.argtypes = [
            C.c_int,
            C.c_void_p, C.c_size_t,
            C.c_void_p, C.c_size_t,
            C.c_int, C.c_int, C.c_int,
            C.POINTER(C.c_int),
        ]
        lib.cc_register_persistent_ccq4_weight.restype = C.c_int
        lib.cc_execute_resident_ccq4_matvec.argtypes = [
            C.c_int, C.c_int, C.c_void_p, C.c_void_p,
        ]
        lib.cc_execute_resident_ccq4_matvec.restype = C.c_int
        lib.cc_release_persistent_weight.argtypes = [C.c_int]
        lib.cc_release_persistent_weight.restype = C.c_int
        lib.cc_release_all_persistent_weights.argtypes = []
        lib.cc_release_all_persistent_weights.restype = C.c_int
        lib.cc_get_persistent_weight_count.argtypes = []
        lib.cc_get_persistent_weight_count.restype = C.c_int
    except AttributeError:
        pass


def last_error(lib: C.CDLL) -> str:
    try:
        raw = lib.cc_get_last_error()
        if raw:
            return raw.decode("utf-8", errors="replace")
    except Exception:
        pass
    return "kein cc_get_last_error verfügbar"


def require_ok(lib: C.CDLL, ok: int, call: str) -> None:
    if ok != 1:
        raise DriverError(f"{call} fehlgeschlagen; last_error={last_error(lib)}")


def require_buffer(ptr: int | None, call: str) -> C.c_void_p:
    if not ptr:
        raise DriverError(f"{call} lieferte NULL")
    return C.c_void_p(ptr)


def f32_array(values: Iterable[float], n: int | None = None) -> C.Array:
    vals = list(values)
    if n is not None and len(vals) != n:
        raise ValueError(f"expected {n} values, got {len(vals)}")
    return (C.c_float * len(vals))(*vals)


def i32_array(values: Iterable[int], n: int | None = None) -> C.Array:
    vals = list(values)
    if n is not None and len(vals) != n:
        raise ValueError(f"expected {n} values, got {len(vals)}")
    return (C.c_int * len(vals))(*vals)


def ptr(obj: C.Array | C.Structure) -> C.c_void_p:
    return C.cast(obj, C.c_void_p)


def quantum_gate(name: str, target: int = 0, control: int = 0, control2: int = 0,
                 params: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> QuantumGate:
    gate = QuantumGate()
    encoded = name.encode("ascii")[:7]
    gate.name = encoded
    gate.target = target
    gate.control = control
    gate.control2 = control2
    gate.arity = 1 if name.upper() in {"H", "X", "Y", "Z", "RX", "RY", "RZ"} else 2
    if name.upper() in {"CCX", "TOFF"}:
        gate.arity = 3
    for i, value in enumerate(params):
        gate.params[i] = float(value)
    return gate


class DriverSession:
    def __init__(self, dll: str | os.PathLike[str] | None = None, gpu: int = 0):
        self.gpu = gpu
        self.lib = load_driver(dll)
        self._initialized = False

    def __enter__(self) -> C.CDLL:
        require_ok(self.lib, self.lib.initialize_gpu(self.gpu), "initialize_gpu")
        self._initialized = True
        return self.lib

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._initialized:
            try:
                self.lib.finish_gpu(self.gpu)
            except Exception:
                pass
            try:
                self.lib.shutdown_gpu(self.gpu)
            except Exception:
                pass
