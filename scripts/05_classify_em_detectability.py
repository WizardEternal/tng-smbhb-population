"""Apply Lin+2026 EM Lomb-Scargle recovery fractions to every TNG merger.

Extends step 04: adds per-system observer-frame ISCO period, survey-window
flags, and expected sinusoidal/sawtooth recovery counts.  Prints the
funnel-stage table that feeds the gap plot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Funnel-stage labels contain Unicode (×, ⁶, M☉); ensure stdout can encode them
# on Windows (where the default console codepage is cp1252).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify TNG mergers by EM survey recoverability."
    )
    parser.add_argument(
        "--input",
        default=str(_REPO_ROOT / "data" / "blackhole_mergers.hdf5"),
        help="Path to input HDF5 file (default: data/blackhole_mergers.hdf5).",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "data" / "processed" / "em_classified.csv"),
        help="Path to output CSV (default: data/processed/em_classified.csv).",
    )
    parser.add_argument(
        "--simulation", default="TNG100-1",
        help="Simulation name tag (default: TNG100-1).",
    )
    parser.add_argument(
        "--hubble-h", type=float, default=0.6774,
        help="Dimensionless Hubble parameter h (default: 0.6774).",
    )
    parser.add_argument(
        "--min-total-mass", type=float, default=1.2e6,
        help="Quality-cut minimum total mass in M_sun (default: 1.2e6).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
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
    print("TNG SMBHB pipeline — Step 05: classify EM detectability")
    print("=" * 60)
    if args.catalog_csv:
        print(f"Input  : {args.catalog_csv} (--catalog-csv)")
        if args.synthetic:
            print("[NOTE] --synthetic ignored because --catalog-csv was provided.")
    else:
        print(f"Input  : {input_path if not args.synthetic else '<synthetic>'}")
    print(f"Output : {output_path}")

    import numpy as np  # noqa: PLC0415

    from tng_smbhb.catalog import catalog_from_arrays, load_tng_hdf5  # noqa: PLC0415
    from tng_smbhb.em_detectability import classify_em_detectability  # noqa: PLC0415
    from tng_smbhb.gw_classification import classify_bands  # noqa: PLC0415
    from tng_smbhb.plotting import compute_funnel_stages  # noqa: PLC0415
    from tng_smbhb.population import derive_population  # noqa: PLC0415

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
            m1_msun=m1, m2_msun=m2, scale_factor=scale_factor,
            simulation="TNG100-1-synthetic", hubble_h=args.hubble_h,
        )
    else:
        print("\nLoading HDF5 ...")
        catalog = load_tng_hdf5(
            input_path, simulation=args.simulation, hubble_h=args.hubble_h,
        )

    print(f"  Loaded {catalog.n_mergers} mergers.")

    pop = derive_population(catalog, min_total_mass_msun=args.min_total_mass)
    gwc = classify_bands(pop)
    emc = classify_em_detectability(pop, gwc.f_isco_hz)

    # Write CSV.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting CSV to {output_path} ...")

    # `band` is assigned from f_isco_observer_hz = f_isco_source_hz / (1+z);
    # the legacy `f_isco_hz` alias is source-frame.  Both are written so a
    # CSV reader can reproduce the band label exactly.
    header = (
        "m1_msun,m2_msun,scale_factor,redshift,chirp_mass_msun,"
        "total_mass_msun,mass_ratio_q,eta,passes_quality_cut,"
        "f_isco_hz,f_isco_source_hz,f_isco_observer_hz,"
        "band,in_pta,in_lisa,in_gap,"
        "p_orb_isco_rest_days,p_orb_isco_obs_days,"
        "in_stripe82,in_ptf,in_lsst,"
        "expected_sin_stripe82,expected_saw_stripe82,"
        "expected_sin_lsst,expected_saw_lsst"
    )
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i in range(catalog.n_mergers):
            parts = [
                f"{catalog.m1_msun[i]:.6e}",
                f"{catalog.m2_msun[i]:.6e}",
                f"{catalog.scale_factor[i]:.8f}",
                f"{catalog.redshift[i]:.8f}",
                f"{pop.chirp_mass_msun[i]:.6e}",
                f"{pop.total_mass_msun[i]:.6e}",
                f"{pop.mass_ratio_q[i]:.8f}",
                f"{pop.eta[i]:.8f}",
                f"{int(pop.passes_quality_cut[i])}",
                f"{gwc.f_isco_hz[i]:.6e}",
                f"{gwc.f_isco_source_hz[i]:.6e}",
                f"{gwc.f_isco_observer_hz[i]:.6e}",
                str(gwc.band[i]),
                f"{int(gwc.in_pta[i])}",
                f"{int(gwc.in_lisa[i])}",
                f"{int(gwc.in_gap[i])}",
                f"{emc.p_orb_isco_rest_days[i]:.6e}",
                f"{emc.p_orb_isco_obs_days[i]:.6e}",
                f"{int(emc.in_stripe82[i])}",
                f"{int(emc.in_ptf[i])}",
                f"{int(emc.in_lsst[i])}",
                f"{emc.expected_sin_stripe82[i]:.4f}",
                f"{emc.expected_saw_stripe82[i]:.4f}",
                f"{emc.expected_sin_lsst[i]:.4f}",
                f"{emc.expected_saw_lsst[i]:.4f}",
            ]
            fh.write(",".join(parts) + "\n")

    # Print funnel table.
    stages = compute_funnel_stages(pop, gwc, emc, survey="stripe82")
    print("\n--- Funnel (Stripe 82) ---")
    prev_count = None
    for s in stages:
        pct = ""
        if prev_count is not None and prev_count > 0:
            pct = f"  ({100.0 * s.count / prev_count:5.1f}% of prev)"
        marker = " (EXP)" if s.is_expected else ""
        print(f"  {s.label:<48s} {s.count:>10,.1f}{marker}{pct}")
        if s.count > 0:
            prev_count = s.count

    print(f"\nDone. CSV written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
