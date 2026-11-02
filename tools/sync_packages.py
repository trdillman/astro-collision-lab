"""Keep the add-on's copies of shared modules byte-identical to the packages.

The packages under ``packages/`` are the source of truth. Run with no arguments after
editing a shared module to copy it into the add-on; run with ``--check`` (as the tests and
CI do) to fail when any copy has drifted.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "extension" / "astro_collision_lab"
SHARED = {
    "packages/astro-initial-conditions/src/astro_ics/galaxy_models.py": "galaxy_models.py",
    "packages/astro-initial-conditions/src/astro_ics/preset_specs.py": "preset_specs.py",
    "packages/warp-cuda-gl/src/warp_cuda_gl/cuda_gl.py": "cuda_gl.py",
    "packages/warp-cuda-gl/src/warp_cuda_gl/device_appearance.py": "device_appearance.py",
}


def drifted():
    return [(ROOT / src, ADDON / dst) for src, dst in SHARED.items()
            if not (ADDON / dst).exists() or (ROOT / src).read_bytes() != (ADDON / dst).read_bytes()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report drift instead of copying")
    args = parser.parse_args(argv)
    pending = drifted()
    if args.check:
        for src, dst in pending:
            print(f"out of sync: {dst.relative_to(ROOT)} != {src.relative_to(ROOT)}")
        return 1 if pending else 0
    for src, dst in pending:
        dst.write_bytes(src.read_bytes())
        print(f"updated {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
