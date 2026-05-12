from __future__ import annotations

import re
from pathlib import Path


ROOT = Path.cwd()
C_FILE = ROOT / "CC_OpenCL.c"
T18 = ROOT / "subqg_driver_tests" / "test_18_subqg_agents_v7_resonance_tracking.py"


def die(msg: str) -> None:
    raise SystemExit(f"[PATCH-ERROR] {msg}")


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".before_native_readback_bridge")
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def find_matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    in_str = False
    esc = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    die("Matching brace not found")
    return -1


def extract_host_get_call(c: str) -> str:
    host_idx = c.find("DLLEXPORT int subqg_simulation_step_host_fields")
    if host_idx < 0:
        die("subqg_simulation_step_host_fields not found")
    body_open = c.find("{", host_idx)
    body_end = find_matching_brace(c, body_open)
    body = c[body_open:body_end+1]

    m = re.search(r"subqg_get_multifield_state\s*\((.*?)\)\s*;", body, flags=re.S)
    if not m:
        die("subqg_get_multifield_state(...) call not found inside host_fields")

    args = " ".join(m.group(1).split())
    return f"subqg_get_multifield_state({args})"


def patch_c() -> None:
    if not C_FILE.exists():
        die(f"C file not found: {C_FILE}")

    c = C_FILE.read_text(encoding="utf-8", errors="replace")
    if "subqg_simulation_step_native_fields_readback" in c:
        print("[C] already patched")
        return

    get_call = extract_host_get_call(c)

    impl_idx = c.find("DLLEXPORT int subqg_simulation_step_native_fields")
    if impl_idx < 0:
        die("native_fields implementation/prototype not found")
    impl_body_open = c.find("{", impl_idx)
    if impl_body_open < 0:
        next_idx = c.find("DLLEXPORT int subqg_simulation_step_native_fields", impl_idx + 1)
        if next_idx < 0:
            die("native_fields implementation body not found")
        impl_idx = next_idx
        impl_body_open = c.find("{", impl_idx)
    impl_body_end = find_matching_brace(c, impl_body_open)

    bridge = f'''

/*
 * Native field-step with explicit host readback.
 *
 * The original subqg_simulation_step_native_fields ABI mutates only the
 * driver's internal OpenCL buffers. That is fast, but Python-side observability
 * remains stale because the host arrays are not rewritten.
 *
 * This bridge preserves the native internal update and then mirrors the new
 * state into the caller-provided host field arrays via the already stabilized
 * subqg_get_multifield_state path used by subqg_simulation_step_host_fields.
 */
DLLEXPORT int subqg_simulation_step_native_fields_readback(
    int gpu_index,
    int cell_count,
    float *energy,
    float *potential,
    float *temperature,
    float *gravity
) {{
    if (cell_count <= 0) {{
        fprintf(stderr,
                "[C] subqg_simulation_step_native_fields_readback: cell_count must be > 0 (got %d).\\n",
                cell_count);
        return 0;
    }}
    if (!energy || !potential || !temperature || !gravity) {{
        fprintf(stderr,
                "[C] subqg_simulation_step_native_fields_readback: NULL host field pointer.\\n");
        return 0;
    }}

    int ok = subqg_simulation_step_native_fields(gpu_index, cell_count);
    if (!ok) {{
        fprintf(stderr,
                "[C] subqg_simulation_step_native_fields_readback: native field step failed.\\n");
        return 0;
    }}

    if (!{get_call}) {{
        fprintf(stderr,
                "[C] subqg_simulation_step_native_fields_readback: subqg_get_multifield_state failed.\\n");
        return 0;
    }}

    return 1;
}}
'''

    backup(C_FILE)
    c = c[:impl_body_end+1] + bridge + c[impl_body_end+1:]

    proto = '''
DLLEXPORT int subqg_simulation_step_native_fields_readback(
    int gpu_index,
    int cell_count,
    float *energy,
    float *potential,
    float *temperature,
    float *gravity
);
'''
    proto_anchor = "DLLEXPORT int subqg_simulation_step_host_fields"
    pidx = c.find(proto_anchor)
    if pidx > 0:
        line_start = c.rfind("\n", 0, pidx) + 1
        before = c[max(0, pidx-700):pidx]
        if "subqg_simulation_step_native_fields_readback" not in before:
            c = c[:line_start] + proto + c[line_start:]

    C_FILE.write_text(c, encoding="utf-8")
    print("[C] patched:", C_FILE)
    print("[C] get call reused:", get_call)


def patch_test18() -> None:
    if not T18.exists():
        print(f"[PY] skipped, test18 not found: {T18}")
        return

    s = T18.read_text(encoding="utf-8", errors="replace")
    if "subqg_simulation_step_native_fields_readback" in s:
        print("[PY] already patched")
        return

    backup(T18)

    if "self.has_native_field_step" in s:
        s = re.sub(
            r"(self\.has_native_field_step\s*=\s*hasattr\(self\.lib,\s*[\"']subqg_simulation_step_native_fields[\"']\)\s*)",
            r"\1\n        self.has_native_field_readback = hasattr(self.lib, \"subqg_simulation_step_native_fields_readback\")",
            s,
            count=1,
        )
    else:
        s = re.sub(
            r"(self\.has_host_step\s*=\s*hasattr\(self\.lib,\s*[\"']subqg_simulation_step_host_fields[\"']\)\s*)",
            r"\1\n        self.has_native_field_readback = hasattr(self.lib, \"subqg_simulation_step_native_fields_readback\")",
            s,
            count=1,
        )

    bind = '''
        if self.has_native_field_readback:
            self.lib.subqg_simulation_step_native_fields_readback.argtypes = [
                C.c_int,   # gpu
                C.c_int,   # cells
                FloatPtr,  # energy
                FloatPtr,  # potential
                FloatPtr,  # temperature
                FloatPtr,  # gravity
            ]
            self.lib.subqg_simulation_step_native_fields_readback.restype = C.c_int

'''
    s = s.replace("        if self.has_host_step:", bind + "        if self.has_host_step:", 1)

    pattern = re.compile(
        r"self\.lib\.subqg_simulation_step_native_fields\s*\(\s*C\.c_int\(cfg\.gpu\)\s*,\s*C\.c_int\(cells\)\s*\)",
        flags=re.S,
    )
    if pattern.search(s):
        repl = (
            "self.lib.subqg_simulation_step_native_fields_readback(\n"
            "                    C.c_int(cfg.gpu),\n"
            "                    C.c_int(cells),\n"
            "                    self.ptr(energy),\n"
            "                    self.ptr(potential),\n"
            "                    self.ptr(temperature),\n"
            "                    self.ptr(gravity),\n"
            "                ) if self.has_native_field_readback else self.lib.subqg_simulation_step_native_fields(\n"
            "                    C.c_int(cfg.gpu), C.c_int(cells)\n"
            "                )"
        )
        s = pattern.sub(repl, s, count=1)
    else:
        print("[PY-WARN] native_fields call pattern not found; C ABI patched, but test18 may need manual binding.")

    s = s.replace(
        '"native_field_step_available": driver.has_native_field_step,',
        '"native_field_step_available": driver.has_native_field_step,\n            "native_field_readback_available": driver.has_native_field_readback,',
        1,
    )

    T18.write_text(s, encoding="utf-8")
    print("[PY] patched:", T18)


def main() -> int:
    patch_c()
    patch_test18()
    print("[PATCH] Done. Rebuild DLL, then run test18/test25.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
