# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added — Phase 2b

- **`bh_details.py`** (`src/tng_smbhb/bh_details.py`): reads the 5.48 GB
  `blackhole_details.hdf5` supplementary file (TNG100-1) to recover true
  primary BH masses. Loads the `/unique` index for O(log N) ID lookup by
  BH particle ID; batch pre-merger mass extraction is ordered by
  `first_index` to exploit sequential HDF5 reads. Extracts the last
  pre-merger record for each surviving BH, replacing the broken `mass_out`
  field with the true primary mass. 100% resolution rate on TNG100-1
  (18,374/18,374 mergers).
- **`scripts/03_fix_bh_masses.py`**: consumes `data/processed/catalog.csv`
  + `blackhole_details.hdf5`, writes `data/processed/catalog_corrected.csv`
  with true primary masses and realistic mass ratios (q median 0.113,
  q 5/95 = 0.001/0.920; previously forced to q = 1 by equal-mass proxy).
  M_tot range: 2.36×10^6 – 1.24×10^10 M_sun.
- **`--catalog-csv` flag** on scripts 04, 05, 06: all three now accept an
  explicit CSV path, enabling the corrected-catalog run
  (`data/processed/catalog_corrected.csv`). Corrected-run outputs directed
  to `outputs/real_corrected/` (6 PNGs); old broken-mass outputs preserved
  in `outputs/real/`.
- **14 new tests** in `tests/test_bh_details.py` covering the `/unique`
  index lookup, batch mass extraction, edge cases, and the 100%-resolution
  invariant. Total pytest suite: **85 passed** (up from 71).

#### Corrected-run funnel vs prior broken run (TNG100-1, 18,374 mergers)

| Stage | Corrected run | Prior broken run (equal-mass proxy) |
|---|---|---|
| All mergers | 18,374 | 18,374 |
| Quality cut | 18,374 (100%) | 18,374 (100%) |
| LISA-band f_ISCO | 5,885 (32.0%) | 9,976 (54.3%) |
| PTA-band f_ISCO | 0 | 0 |
| Stripe 82 window (200–1100 d) | 0 | 0 |
| Sinusoidal-recoverable | 0 | 0 |
| Sawtooth-recoverable | 0 | 0 |

The prior run inflated the LISA count (9,976 vs 5,885) because the
equal-mass proxy set q = 1 for all systems, boosting chirp masses. The
corrected run uses realistic mass ratios and is physically meaningful.
Stripe 82 remains zero — see Known limitations below.

### Added — Phase 2c

- **CI/tooling hardening** (`.github/workflows/ci.yml`, `pyproject.toml`,
  `.pre-commit-config.yaml`, `environment.yml`): upper bounds on all
  runtime deps verified; CI matrix `["3.11", "3.12"]` consistent with
  `requires-python = ">=3.11"`; pre-commit hooks (ruff + black + mypy)
  added; conda `environment.yml` reflects pyproject pins. Sibling repos
  (smbhb-inspiral, stripe82-reference) aligned.

### Added — Phase 1c

- **README gap-plot embed** (`README.md`): canonical synthetic seed=42
  Stripe 82 gap plot (`outputs/synthetic/gap_plot_stripe82.png`) now embedded
  as the flagship figure near the top, between the three-repo narrative
  and the caveats section. The gap plot is the portfolio's visual
  punchline per EXECUTION_PLAN.md L10 and SESSION_HANDOFF.md lines 121-135.

### Fixed

- **Redshift bug in `gw_classification.py`** (`src/tng_smbhb/gw_classification.py`):
  Band classification was using source-frame f_ISCO. GW detectors (PTA, LISA)
  observe in the Earth frame, so band assignment must use
  `f_obs = f_ISCO_source / (1 + z)`. Added `compute_f_isco_observer()`,
  updated `classify_bands()` to use the observer-frame frequency for band
  assignment, and added `f_isco_source_hz` / `f_isco_observer_hz` dual fields
  to `GWClassification` (with `f_isco_hz` kept as a backward-compat alias for
  `f_isco_source_hz`, used by `em_detectability.py`). Updated module docstring
  to make the frame convention explicit. Added 7 new tests in
  `tests/test_gw_classification.py` (total: 15, up from 8): observer-frame
  regression at z=2, redshift-dependence verification, `f_obs <= f_source`
  invariant. Synthetic funnel numbers change: LISA 1593 → 1377, PTA 0 → 1.
- **`_vendored_em_detectability.py:41` stale doctest** (`src/tng_smbhb/_vendored_em_detectability.py`):
  Fixed import example from `smbhb_inspiral.em_detectability` to
  `tng_smbhb._vendored_em_detectability`.
- **Orphan output files reorganized** (`outputs/`):
  `em_classified_synth.csv` and `gw_classified_synth.csv` moved to
  `outputs/synthetic/`. Real-data PNGs moved to `outputs/real/`. Updated
  `scripts/06_generate_plots.py` to auto-select `outputs/synthetic/` or
  `outputs/real/` based on data source; `--outdir` overrides.
- **README real-data clarification** (`README.md`):
  Added "Real data vs synthetic" paragraph in the Caveats section explaining
  that the canonical Gate-2 gap plot uses synthetic seed=42 data, that the
  real TNG100-1 run is a pipeline sanity check only (0 EM hits due to the
  `mass_out` bug equal-mass proxy), and that the proper fix is Phase 2b.
  Replaced TBD Results table with actual numbers for both synthetic and real
  runs (observer-frame band counts).

### Added — Phase 1b core

Package modules (`src/tng_smbhb/`):

- `catalog.py` — `TNGMergerCatalog` frozen dataclass; `load_tng_hdf5` with
  unit conversion (TNG internal `10^10 M_sun / h` → `M_sun`; scale factor
  `a` → redshift `z = 1/a − 1`); `catalog_from_arrays` constructor for
  synthetic data.
- `population.py` — `TNGPopulation` dataclass; vectorized chirp mass,
  total mass, mass ratio, symmetric mass ratio; quality-cut mask
  excluding seed-mass systems (`M_tot < 1.2 × 10^6 M_sun`).
- `gw_classification.py` — `GWClassification` dataclass; per-system
  `f_ISCO`; band assignment `"pta"` / `"lisa"` / `"gap"` / `"neither"` with
  locked edges (PTA: `1e-9`–`1e-7 Hz`; LISA: `1e-4`–`1e-1 Hz`).
- `em_detectability.py` — `EMClassification` dataclass; applies vendored
  Lin, Charisi & Haiman 2026 (ApJ 997, 316) recovery fractions
  (sin 45/24/23 %, saw 9/1/1 % for PTF-like / idealized / LSST-like);
  observer-frame ISCO-period survey-window classification.
- `plotting.py` — `make_gap_plot` (single-survey), `make_gap_plot_dual_survey`,
  `make_mass_distribution_plot`, `make_redshift_mass_plot`;
  `compute_funnel_stages` returns the 7-stage funnel fingerprint.
- `_vendored_constants.py`, `_vendored_physics.py`,
  `_vendored_em_detectability.py` — verbatim copies of the sibling
  `smbhb-inspiral` v0.1.0 modules with a banner comment (no pip
  cross-import; decision L2 in EXECUTION_PLAN.md).

CLI pipeline (`scripts/`, thin wrappers, each with `--synthetic` fallback
where applicable):

- `01_download_mergers.py` — verify `data/blackhole_mergers.hdf5`
  presence; print download instructions if missing.
- `02_parse_catalog.py` — HDF5 → `data/processed/catalog.csv`.
- `03_host_match.py` — Phase 2b stub (exits 2).
- `04_classify_gw_bands.py` — adds GW-band columns → `gw_classified.csv`.
- `05_classify_em_detectability.py` — adds EM columns → `em_classified.csv`;
  prints the funnel table.
- `06_generate_plots.py` — renders all six figures including the gap
  plot.

Tests (`tests/`, 63 passing):

- `test_catalog.py` (19), `test_population.py` (15),
  `test_gw_classification.py` (8), `test_em_detectability.py` (7),
  `test_plotting.py` (12), `test_integration.py` (2), plus
  `conftest.py::synthetic_tng_catalog` deterministic fixture
  (seed = 42, 500 mergers).

Documentation:

- `README.md` with softening-scale caveat section leading the body
  (TNG mergers are at the gravitational softening length, not physical
  coalescence; final-parsec delay is unconstrained).
- `data/PROVENANCE.md`, `CITATION.cff`, `LICENSE` (MIT).

### Verified

- **Gate 2 (multi-messenger gap plot):** renders cleanly on the seed=42
  synthetic catalog (5000 mergers). Funnel collapse:
  `5000 → 4085 (quality) → 1377 LISA → 9 (Stripe 82 window) →
  4.1 (sinusoidal-recoverable) → 0.8 (sawtooth-recoverable)`.
  (Pre-observer-frame-correction runs reported 1593 LISA; see the
  "Fixed" section below for why that dropped to 1377.)
  The dramatic multi-order-of-magnitude drop from raw mergers to
  sawtooth-recoverable is the intended visual punchline of the portfolio.
  Regeneration: `PYTHONPATH=src python scripts/06_generate_plots.py --synthetic`.
- Full pytest suite: `63 passed` on Windows / Python 3.12.3.

### Known limitations

- **TNG100-1 mass ceiling / Stripe 82 structural zero (Phase 2b verified).**
  The heaviest system in TNG100-1 is M_tot = 1.24×10^10 M_sun. At ISCO
  and z = 0 this gives P_orb_obs ≈ 65 days, well below the Stripe 82
  window floor of 200 days. The window requires M_tot ≥ 3.8×10^10 M_sun,
  which TNG100-1's box volume simply does not contain. The Stripe 82 = 0
  result is therefore a **structural limitation of the simulation volume**,
  not a data bug. Proper fixes (Phase 3): (a) use TNG300-1 to sample the
  ultra-massive tail, or (b) evaluate P_obs during the hardening/inspiral
  phase rather than at ISCO, where systems spend far more time at longer
  periods.
- EM detectability is evaluated at `f_ISCO`, not time-weighted over the
  inspiral — Phase 1b convention per `EXECUTION_PLAN.md` §5.3. Real
  inspirals spend most of their time at longer periods; proper
  time-weighted classification is Phase 3 material.
- Real TNG100-1 Gate-2 headline numbers now come from the corrected run
  (`outputs/real_corrected/`). The old broken-mass outputs are preserved
  in `outputs/real/` for reference.
