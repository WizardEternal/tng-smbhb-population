"""Load TNG HDF5, derive population quantities, and write catalog.csv."""
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
        description="Parse TNG HDF5 merger file into a population CSV."
    )
    parser.add_argument(
        "--input",
        default=str(_REPO_ROOT / "data" / "blackhole_mergers.hdf5"),
        help="Path to input HDF5 file (default: data/blackhole_mergers.hdf5)",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "data" / "processed" / "catalog.csv"),
        help="Path to output CSV (default: data/processed/catalog.csv)",
    )
    parser.add_argument(
        "--simulation",
        default="TNG100-1",
        help="Simulation name tag (default: TNG100-1)",
    )
    parser.add_argument(
        "--hubble-h",
        type=float,
        default=0.6774,
        help="Dimensionless Hubble parameter h (default: 0.6774)",
    )
    parser.add_argument(
        "--min-total-mass",
        type=float,
        default=1.2e6,
        help="Quality-cut minimum total mass in M_sun (default: 1.2e6)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    print("=" * 60)
    print("TNG SMBHB pipeline — Step 02: parse catalog")
    print("=" * 60)
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Simulation : {args.simulation}  h = {args.hubble_h}")
    print(f"Quality cut: M_tot > {args.min_total_mass:.2e} M_sun")

    if not input_path.exists():
        print(f"\n[ERROR] HDF5 file not found: {input_path}")
        print("Run step 01 first to obtain the data file.")
        return 1

    # Imports deferred so missing-file error prints cleanly
    import numpy as np  # noqa: PLC0415

    from tng_smbhb.catalog import load_tng_hdf5  # noqa: PLC0415
    from tng_smbhb.population import derive_population  # noqa: PLC0415

    print("\nLoading HDF5 …")
    catalog = load_tng_hdf5(
        input_path,
        simulation=args.simulation,
        hubble_h=args.hubble_h,
    )
    print(f"  Loaded {catalog.n_mergers} raw merger events.")

    print("Deriving population quantities …")
    pop = derive_population(catalog, min_total_mass_msun=args.min_total_mass)

    # Build column arrays
    m1 = catalog.m1_msun
    m2 = catalog.m2_msun
    a = catalog.scale_factor
    z = catalog.redshift
    m_chirp = pop.chirp_mass_msun
    m_tot = pop.total_mass_msun
    q = pop.mass_ratio_q
    eta = pop.eta
    passes = pop.passes_quality_cut

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting CSV to {output_path} …")

    header = "m1_msun,m2_msun,scale_factor,redshift,chirp_mass_msun,total_mass_msun,mass_ratio_q,eta,passes_quality_cut"
    rows = np.column_stack([m1, m2, a, z, m_chirp, m_tot, q, eta, passes.astype(np.int32)])
    np.savetxt(
        output_path,
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt=[
            "%.6e", "%.6e", "%.8f", "%.8f",
            "%.6e", "%.6e", "%.8f", "%.8f", "%d",
        ],
    )

    # Summary
    n_passing = int(np.sum(passes))
    m_tot_pass = m_tot[passes]
    z_all = z

    print("\n--- Summary ---")
    print(f"  N total mergers        : {catalog.n_mergers}")
    print(f"  N passing quality cut  : {n_passing}")
    print(f"  Min M_tot (all)        : {m_tot.min():.3e} M_sun")
    print(f"  Max M_tot (all)        : {m_tot.max():.3e} M_sun")
    if n_passing > 0:
        print(f"  Min M_tot (passing)    : {m_tot_pass.min():.3e} M_sun")
        print(f"  Max M_tot (passing)    : {m_tot_pass.max():.3e} M_sun")
    print(f"  Redshift range         : z = {z_all.min():.3f} – {z_all.max():.3f}")
    print(f"\nDone. CSV written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
