"""Classify each TNG SMBHB merger by GW band (PTA/LISA/gap/neither).

Loads the raw HDF5 catalog (or a synthetic fallback via --synthetic), derives
population quantities, computes f_ISCO per system, assigns each system to its
GW band, and writes an augmented CSV to data/processed/gw_classified.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure stdout can encode non-ASCII diagnostic output on Windows cp1252.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# Make src/ importable when running directly without pip install -e .
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify TNG SMBHB mergers by GW band."
    )
    parser.add_argument(
        "--input",
        default=str(_REPO_ROOT / "data" / "blackhole_mergers.hdf5"),
        help="Path to input HDF5 file (default: data/blackhole_mergers.hdf5).",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "data" / "processed" / "gw_classified.csv"),
        help="Path to output CSV (default: data/processed/gw_classified.csv).",
    )
    parser.add_argument(
        "--simulation",
        default="TNG100-1",
        help="Simulation name tag (default: TNG100-1).",
    )
    parser.add_argument(
        "--hubble-h",
        type=float,
        default=0.6774,
        help="Dimensionless Hubble parameter h (default: 0.6774).",
    )
    parser.add_argument(
        "--min-total-mass",
        type=float,
        default=1.2e6,
        help="Quality-cut minimum total mass in M_sun (default: 1.2e6).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a 5000-merger synthetic catalog (seed=42) instead of HDF5.",
    )
    parser.add_argument(
        "--catalog-csv",
        default=None,
        help="Path to a corrected catalog CSV (e.g. catalog_corrected.csv). "
             "If provided, short-circuits the HDF5 load. "
             "Columns 0/1/2 must be m1_msun, m2_msun, scale_factor.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    print("=" * 60)
    print("TNG SMBHB pipeline — Step 04: classify GW bands")
    print("=" * 60)
    if args.catalog_csv:
        print(f"Input  : {args.catalog_csv} (--catalog-csv)")
        if args.synthetic:
            print("[NOTE] --synthetic ignored because --catalog-csv was provided.")
    else:
        print(f"Input  : {input_path if not args.synthetic else '<synthetic>'}")
    print(f"Output : {output_path}")

    # Imports deferred so CLI errors print cleanly before importing heavy deps.
    import numpy as np  # noqa: PLC0415

    from tng_smbhb.catalog import catalog_from_arrays, load_tng_hdf5  # noqa: PLC0415
    from tng_smbhb.gw_classification import classify_bands  # noqa: PLC0415
    from tng_smbhb.population import derive_population  # noqa: PLC0415

    # Load or synthesize catalog.
    if args.catalog_csv is not None:
        csv_path = Path(args.catalog_csv).resolve()
        print(f"\nLoading corrected catalog CSV: {csv_path} ...")
        data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
        catalog = catalog_from_arrays(
            m1_msun=data[:, 0],
            m2_msun=data[:, 1],
            scale_factor=data[:, 2],
            simulation=args.simulation,
            hubble_h=args.hubble_h,
        )
    elif args.synthetic or not input_path.exists():
        if not args.synthetic:
            print(
                f"\n[WARN] HDF5 not found at {input_path}; "
                f"falling back to synthetic catalog."
            )
        print("Generating synthetic catalog (seed=42, 5000 mergers) ...")
        rng = np.random.default_rng(42)
        n = 5000
        m1 = 10 ** rng.uniform(5.0, 10.0, n)
        m2 = m1 * rng.uniform(0.1, 1.0, n)
        scale_factor = rng.uniform(0.3, 1.0, n)
        catalog = catalog_from_arrays(
            m1_msun=m1,
            m2_msun=m2,
            scale_factor=scale_factor,
            simulation="TNG100-1-synthetic",
            hubble_h=args.hubble_h,
        )
    else:
        print("\nLoading HDF5 ...")
        catalog = load_tng_hdf5(
            input_path,
            simulation=args.simulation,
            hubble_h=args.hubble_h,
        )

    print(f"  Loaded {catalog.n_mergers} merger events.")

    print("Deriving population ...")
    pop = derive_population(catalog, min_total_mass_msun=args.min_total_mass)

    print("Classifying GW bands ...")
    gwc = classify_bands(pop)

    # Write CSV.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting CSV to {output_path} ...")

    header = (
        "m1_msun,m2_msun,scale_factor,redshift,chirp_mass_msun,"
        "total_mass_msun,mass_ratio_q,eta,passes_quality_cut,"
        "f_isco_hz,band,in_pta,in_lisa,in_gap"
    )
    rows_numeric = np.column_stack(
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
            gwc.f_isco_hz,
            # band string column appended via object-array concat below
            gwc.in_pta.astype(np.int32),
            gwc.in_lisa.astype(np.int32),
            gwc.in_gap.astype(np.int32),
        ]
    )
    # We need to inject the string band column; easiest: write by row.
    fmt_numeric_per_col = [
        "%.6e", "%.6e", "%.8f", "%.8f",
        "%.6e", "%.6e", "%.8f", "%.8f",
        "%d", "%.6e",
        "%d", "%d", "%d",
    ]
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i in range(catalog.n_mergers):
            numeric_cols = rows_numeric[i].tolist()
            pre = numeric_cols[:10]  # up to and including f_isco_hz
            post = numeric_cols[10:]  # in_pta, in_lisa, in_gap
            parts = [fmt_numeric_per_col[j] % pre[j] for j in range(10)]
            parts.append(str(gwc.band[i]))
            parts += [fmt_numeric_per_col[10 + j] % post[j] for j in range(3)]
            fh.write(",".join(parts) + "\n")

    # Summary.
    print("\n--- GW band counts ---")
    print(f"  PTA     : {gwc.n_pta:>6d}")
    print(f"  LISA    : {gwc.n_lisa:>6d}")
    print(f"  gap     : {gwc.n_gap:>6d}")
    print(f"  neither : {gwc.n_neither:>6d}")
    print(f"  total   : {catalog.n_mergers:>6d}")
    print(f"\nDone. CSV written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
