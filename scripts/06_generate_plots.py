"""Generate all Phase-1b figures, including the multi-messenger gap plot (figure #9).

Uses the real TNG100-1 HDF5 if present at data/blackhole_mergers.hdf5, or a
5000-merger deterministic synthetic catalog (seed=42) via --synthetic or as an
automatic fallback.  Writes these files under outputs/:

    gap_plot_stripe82.png          (dark theme, primary)
    gap_plot_stripe82_light.png    (light theme, READMEs)
    gap_plot_lsst.png              (LSST variant)
    gap_plot_dual.png              (2-panel Stripe 82 + LSST)
    mass_distribution.png          (M_tot + M_chirp histograms)
    redshift_mass_scatter.png      (z vs log10 M_tot, colored by GW band)
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
        description="Generate all Phase-1b figures, including the gap plot."
    )
    parser.add_argument(
        "--input",
        default=str(_REPO_ROOT / "data" / "blackhole_mergers.hdf5"),
        help="Path to input HDF5 file (default: data/blackhole_mergers.hdf5).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Directory for generated figures. "
            "Defaults to outputs/synthetic/ when --synthetic is given, "
            "outputs/real/ otherwise."
        ),
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
        help="Force synthetic catalog even if HDF5 is present.",
    )
    parser.add_argument(
        "--catalog-csv",
        default=None,
        help="Path to a corrected catalog CSV (e.g. catalog_corrected.csv). "
             "If provided, short-circuits the HDF5 load. "
             "Columns 0/1/2 must be m1_msun, m2_msun, scale_factor. "
             "Default outdir becomes outputs/real_corrected/ when used.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()

    # If the user did not specify --outdir, auto-select based on data source.
    # --synthetic        → outputs/synthetic/
    # --catalog-csv      → outputs/real_corrected/
    # real HDF5          → outputs/real/
    if args.outdir is None:
        if args.catalog_csv is not None:
            default_subdir = "real_corrected"
        elif args.synthetic or not input_path.exists():
            default_subdir = "synthetic"
        else:
            default_subdir = "real"
        outdir = (_REPO_ROOT / "outputs" / default_subdir).resolve()
    else:
        outdir = Path(args.outdir).resolve()

    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TNG SMBHB pipeline — Step 06: generate plots")
    print("=" * 60)

    # Force non-interactive backend BEFORE importing pyplot anywhere.
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")

    import numpy as np  # noqa: PLC0415

    from tng_smbhb.catalog import catalog_from_arrays, load_tng_hdf5  # noqa: PLC0415
    from tng_smbhb.em_detectability import classify_em_detectability  # noqa: PLC0415
    from tng_smbhb.gw_classification import classify_bands  # noqa: PLC0415
    from tng_smbhb.plotting import (  # noqa: PLC0415
        compute_funnel_stages,
        make_gap_plot,
        make_gap_plot_dual_survey,
        make_mass_distribution_plot,
        make_redshift_mass_plot,
    )
    from tng_smbhb.population import derive_population  # noqa: PLC0415

    if args.catalog_csv is not None:
        csv_path = Path(args.catalog_csv).resolve()
        print(f"\nLoading corrected catalog CSV: {csv_path} ...")
        if args.synthetic:
            print("[NOTE] --synthetic ignored because --catalog-csv was provided.")
        data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
        catalog = catalog_from_arrays(
            m1_msun=data[:, 0],
            m2_msun=data[:, 1],
            scale_factor=data[:, 2],
            simulation=args.simulation,
            hubble_h=args.hubble_h,
        )
    else:
        use_synthetic = args.synthetic or not input_path.exists()
        if use_synthetic and not args.synthetic:
            print(f"\n[WARN] HDF5 not found at {input_path}; using synthetic catalog.")

        if use_synthetic:
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
            print(f"\nLoading HDF5: {input_path}")
            catalog = load_tng_hdf5(
                input_path, simulation=args.simulation, hubble_h=args.hubble_h,
            )

    print(f"  Loaded {catalog.n_mergers} mergers ({catalog.simulation}).")

    pop = derive_population(catalog, min_total_mass_msun=args.min_total_mass)
    gwc = classify_bands(pop)
    emc = classify_em_detectability(pop, gwc.f_isco_hz)

    # Print the Stripe 82 funnel for the record:
    stages = compute_funnel_stages(pop, gwc, emc, survey="stripe82")
    print("\n--- Stripe 82 funnel ---")
    for s in stages:
        marker = "(EXP)" if s.is_expected else "     "
        print(f"  {s.label:<48s} {s.count:>10,.1f}  {marker}")

    # Emit the figures.
    print("\nRendering figures ...")

    outputs: list[tuple[str, Path]] = []

    p = make_gap_plot(
        pop, gwc, emc,
        survey="stripe82",
        outpath=outdir / "gap_plot_stripe82.png",
        theme="dark",
    )
    outputs.append(("gap (Stripe 82, dark) ", p))

    p = make_gap_plot(
        pop, gwc, emc,
        survey="stripe82",
        outpath=outdir / "gap_plot_stripe82_light.png",
        theme="light",
    )
    outputs.append(("gap (Stripe 82, light)", p))

    p = make_gap_plot(
        pop, gwc, emc,
        survey="lsst",
        outpath=outdir / "gap_plot_lsst.png",
        theme="dark",
    )
    outputs.append(("gap (LSST, dark)      ", p))

    p = make_gap_plot_dual_survey(
        pop, gwc, emc,
        outpath=outdir / "gap_plot_dual.png",
        theme="dark",
    )
    outputs.append(("gap (dual-survey)     ", p))

    p = make_mass_distribution_plot(pop, outpath=outdir / "mass_distribution.png")
    outputs.append(("mass distribution     ", p))

    p = make_redshift_mass_plot(
        pop, gwc, outpath=outdir / "redshift_mass_scatter.png"
    )
    outputs.append(("z-vs-mass scatter     ", p))

    print("\nWrote:")
    for label, path in outputs:
        try:
            size_kb = path.stat().st_size / 1024.0
            print(f"  {label}  {path}  ({size_kb:.1f} kB)")
        except OSError:
            print(f"  {label}  {path}  (missing)")

    print(f"\nDone. All figures in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
