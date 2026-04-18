"""Fix the TNG ``mass_out`` bug using ``blackhole_details.hdf5`` pre-merger masses.

Reads the mergers file's ``id_out`` and ``time`` arrays, looks up each surviving
BH's last record strictly before the merger time in the details file, and
writes a corrected catalog CSV with the true primary mass replacing the
Phase 1b equal-mass proxy.

Inputs
------
data/blackhole_mergers.hdf5
data/blackhole_details.hdf5

Output
------
data/processed/catalog_corrected.csv  (same schema as catalog.csv)

This script does NOT modify catalog.csv — it emits a sibling file so the
equal-mass-proxy run remains reproducible for comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# UTF-8 stdout on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fix TNG mass_out bug via blackhole_details.hdf5 pre-merger lookup."
    )
    p.add_argument(
        "--mergers",
        default=str(_REPO_ROOT / "data" / "blackhole_mergers.hdf5"),
    )
    p.add_argument(
        "--details",
        default=str(_REPO_ROOT / "data" / "blackhole_details.hdf5"),
    )
    p.add_argument(
        "--output",
        default=str(_REPO_ROOT / "data" / "processed" / "catalog_corrected.csv"),
    )
    p.add_argument("--simulation", default="TNG100-1")
    p.add_argument("--hubble-h", type=float, default=0.6774)
    p.add_argument("--min-total-mass", type=float, default=1.2e6)
    args = p.parse_args()

    mergers_path = Path(args.mergers).resolve()
    details_path = Path(args.details).resolve()
    output_path = Path(args.output).resolve()

    print("=" * 60)
    print("TNG SMBHB pipeline — Step 03: fix mass_out via blackhole_details")
    print("=" * 60)
    print(f"Mergers : {mergers_path}")
    print(f"Details : {details_path}")
    print(f"Output  : {output_path}")

    for label, pth in (("mergers", mergers_path), ("details", details_path)):
        if not pth.exists():
            print(f"\n[ERROR] {label} file not found: {pth}")
            return 1

    import h5py  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from tng_smbhb.bh_details import (  # noqa: PLC0415
        batch_lookup_premerger_masses,
        load_details_index,
    )
    from tng_smbhb.catalog import catalog_from_arrays  # noqa: PLC0415
    from tng_smbhb.population import derive_population  # noqa: PLC0415

    print("\n[1/5] Reading mergers HDF5 (id_out, mass_in, time) ...")
    with h5py.File(mergers_path, "r") as f:
        id_out = np.asarray(f["id_out"], dtype=np.uint64)
        mass_in_raw = np.asarray(f["mass_in"], dtype=np.float64)
        time_a = np.asarray(f["time"], dtype=np.float64)
    n_mergers = int(id_out.size)
    print(f"   {n_mergers} mergers loaded.")

    print("\n[2/5] Loading details /unique index ...")
    index = load_details_index(details_path)
    print(f"   {index.num_blackholes} unique BHs indexed.")

    print("\n[3/5] Looking up pre-merger primary mass for each merger ...")
    m1_primary_msun, found = batch_lookup_premerger_masses(
        index,
        bh_ids=id_out,
        merger_times=time_a,
        hubble_h=args.hubble_h,
    )
    n_found = int(found.sum())
    n_missing = n_mergers - n_found
    print(f"   Resolved {n_found}/{n_mergers} primary masses "
          f"({n_missing} missing; fallback to mass_in).")

    # Secondary (already correct) in M_sun
    scale = 1.0e10 / args.hubble_h
    m_secondary_msun = mass_in_raw * scale

    # Primary: corrected where available; fall back to mass_in (equal-mass proxy)
    m1_out = np.where(found, m1_primary_msun, m_secondary_msun)
    m2_out = m_secondary_msun

    # Validity mask (finite, positive, a > 0)
    valid = (
        np.isfinite(m1_out) & (m1_out > 0.0)
        & np.isfinite(m2_out) & (m2_out > 0.0)
        & np.isfinite(time_a) & (time_a > 0.0)
    )
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f"   Dropping {n_dropped} invalid rows.")

    m1_out = m1_out[valid]
    m2_out = m2_out[valid]
    a_valid = time_a[valid]

    print("\n[4/5] Building corrected catalog + population quantities ...")
    catalog = catalog_from_arrays(
        m1_msun=m1_out,
        m2_msun=m2_out,
        scale_factor=a_valid,
        simulation=args.simulation + "-corrected",
        hubble_h=args.hubble_h,
    )
    pop = derive_population(catalog, min_total_mass_msun=args.min_total_mass)

    # --- Diagnostics: mass-ratio distribution ---
    q = pop.mass_ratio_q
    print(f"   q median = {np.median(q):.3f}   "
          f"q 5/95 = {np.percentile(q, 5):.3f}/{np.percentile(q, 95):.3f}")
    n_eq = int(np.sum(q > 0.999))
    print(f"   q > 0.999 (equal-mass fallback or genuine): {n_eq} "
          f"(~{100.0 * n_eq / max(q.size, 1):.1f}%)")
    print(f"   M_tot min/median/max = "
          f"{pop.total_mass_msun.min():.2e} / "
          f"{np.median(pop.total_mass_msun):.2e} / "
          f"{pop.total_mass_msun.max():.2e} M_sun")

    print(f"\n[5/5] Writing {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "m1_msun,m2_msun,scale_factor,redshift,chirp_mass_msun,"
        "total_mass_msun,mass_ratio_q,eta,passes_quality_cut"
    )
    rows = np.column_stack(
        [
            catalog.m1_msun,
            catalog.m2_msun,
            catalog.scale_factor,
            catalog.redshift,
            pop.chirp_mass_msun,
            pop.total_mass_msun,
            pop.mass_ratio_q,
            pop.eta,
            pop.passes_quality_cut.astype(np.int32),
        ]
    )
    np.savetxt(
        output_path,
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%.6e", "%.6e", "%.8f", "%.8f", "%.6e", "%.6e",
             "%.8f", "%.8f", "%d"],
    )

    n_passing = int(pop.passes_quality_cut.sum())
    print(f"\n--- Summary ---")
    print(f"  N mergers kept         : {catalog.n_mergers}")
    print(f"  N passing quality cut  : {n_passing}")
    print(f"  Mass-corrected rows    : {n_found} ({100.0 * n_found / n_mergers:.1f}%)")
    print(f"\nDone. Corrected CSV at: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
