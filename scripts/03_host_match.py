"""TNG subhalo host cross-matching — SUPERSEDED by 03_fix_bh_masses.py.

The Phase 2b plan originally called for O(10^4) rate-limited TNG API calls
against the ``SubhaloBHMass`` field to recover the missing primary masses.
That path was abandoned in favor of reading the public supplementary catalog
``blackhole_details.hdf5`` (5.2 GB), whose per-timestep ``mass`` field is the
correct physical BH mass and which supports O(1) per-BH lookup via the
``/unique`` subgroup.

Use ``03_fix_bh_masses.py`` instead.  Running this script exits 2.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def main() -> int:
    print(
        "This script is superseded by 03_fix_bh_masses.py, which uses the\n"
        "blackhole_details.hdf5 supplementary catalog instead of the TNG API.\n"
        "Exiting with code 2."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
