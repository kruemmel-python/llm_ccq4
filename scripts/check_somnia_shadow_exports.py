from __future__ import annotations

import argparse
from ctypes import CDLL

REQUIRED = [
    "shadow_init",
    "shadow_inject_pulse",
    "shadow_start_loop",
    "shadow_cycle",
    "shadow_read_signal",
    "shadow_set_abort_flag",
    "shadow_checkpoint_save",
    "shadow_checkpoint_load",
]
OPTIONAL_ENTERPRISE = [
    "shadow_inject_pulse_packet",
    "shadow_read_signal_packet",
]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dll")
    args = parser.parse_args()

    dll = CDLL(args.dll)
    missing = []
    for name in REQUIRED + OPTIONAL_ENTERPRISE:
        ok = hasattr(dll, name)
        print(f"{name:32} {'OK' if ok else 'MISSING'}")
        if name in REQUIRED and not ok:
            missing.append(name)

    return 1 if missing else 0

if __name__ == "__main__":
    raise SystemExit(main())
