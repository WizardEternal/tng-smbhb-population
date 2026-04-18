"""Check for TNG blackhole_mergers.hdf5 and print download instructions if missing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Unicode arrows and other non-ASCII symbols appear in the user-facing help
# text; ensure stdout can encode them on Windows (default codepage cp1252).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# Make src/ importable when running directly without pip install -e .
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _hdf5_top_level_keys(path: Path) -> list[str]:
    """Return top-level HDF5 keys without importing h5py at module level."""
    import h5py  # noqa: PLC0415

    with h5py.File(path, "r") as f:
        return list(f.keys())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for TNG blackhole_mergers.hdf5 and print instructions."
    )
    parser.add_argument(
        "--path",
        default=str(_REPO_ROOT / "data" / "blackhole_mergers.hdf5"),
        help="Path to the HDF5 merger file (default: data/blackhole_mergers.hdf5)",
    )
    args = parser.parse_args()

    hdf5_path = Path(args.path).resolve()

    print("=" * 60)
    print("TNG SMBHB pipeline — Step 01: data acquisition check")
    print("=" * 60)
    print(f"Looking for: {hdf5_path}")

    if not hdf5_path.exists():
        print("\n[MISSING] File not found.\n")
        print("Download instructions:")
        print("  1. Go to https://www.tng-project.org and log in (free account).")
        print("  2. Navigate to: Simulations → TNG100-1.")
        print("  3. In the left panel, click 'Merger Tree Data'.")
        print("  4. Download 'blackhole_mergers.hdf5'.")
        print(f"  5. Save the file to: {hdf5_path.parent}/")
        print("  6. Re-run this script to verify.")
        return 1

    size_mb = hdf5_path.stat().st_size / 1024 / 1024
    print(f"\n[FOUND] File size: {size_mb:.2f} MB")

    print("\nTop-level HDF5 structure:")
    try:
        keys = _hdf5_top_level_keys(hdf5_path)
        if keys:
            for key in keys:
                print(f"  /{key}")
        else:
            print("  (no top-level groups or datasets)")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not read HDF5 structure: {exc}")

    print("\nFile is present and readable. Proceed to step 02.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
