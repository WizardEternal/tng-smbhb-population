# tng-smbhb-population

This repo takes the IllustrisTNG SMBH merger catalog and produces the multi-messenger gap plot: the funnel from ~18,374 TNG100-1 black-hole merger events down to the handful of systems that a Lomb-Scargle search on Stripe 82 data could plausibly recover. The funnel is what I was trying to produce, and the rest of the code is what it takes to get there.

Fair warning up front: this isn't a novel population synthesis. IllustrisTNG BH "mergers" trigger when two BH particles come within the gravitational softening length, which is roughly 0.7 kpc for TNG100-1. That's a numerical event in the simulation, not a physical parsec-scale coalescence where GW emission starts to matter. The catalog I'm working with contains progenitor pairs that *might* become GW sources after an uncertain 100 Myr to multi-Gyr delay. I'm not claiming they will. What I built is a pipeline that asks: IF each of these pairs eventually coalesces, which GW band carries the signal, and does its observer-frame orbital period fall inside a current EM survey window? The multi-order-of-magnitude drop from raw TNG mergers to survey-recoverable candidates is the thing I wanted to quantify.

## Background

Supermassive black hole binaries (SMBHBs) form when two galaxies merge and their central BHs sink toward the center of the combined system via dynamical friction. From there, the orbit tightens through stellar scattering and gas torques until GW emission takes over as the dominant energy-loss channel. The problem is that "takes over" requires shrinking the separation below roughly 1 pc, and getting there is theoretically hard (the final parsec problem).

Once a pair does reach GW-emitting separations, different mass regimes fall in different detector bands. Pulsar Timing Arrays are sensitive to GW frequencies of roughly 1-100 nHz, which corresponds to systems with M_tot > 10^8 M_sun (Rajagopal and Romani 1995; Sesana et al. 2004). LISA, targeting ~0.1-100 mHz, will probe M_tot 10^5-10^7 M_sun systems when it launches in the mid-2030s. The two populations are mass-disjoint: there's a gap between them around 10^7-10^8 M_sun where neither detector is sensitive.

EM searches offer an independent, currently operational path. Periodic optical variability, hypothetically driven by Doppler-boosted accretion or orbital modulation of the disk, has been hunted in time-domain surveys. Charisi et al. 2016 ran a similar search in PTF data and found 50 candidates; Graham et al. 2015 did it in CRTS with 111 candidates. But Lomb-Scargle applied to light curves contaminated by DRW (damped random walk) noise recovers only about 45% of injected sinusoidal signals and roughly 9% of sawtooth signals (Lin, Charisi and Haiman 2026). That's the other side of the gap: even if a system is in the right period window, the LS detection probability is low.

The gap this repo quantifies is the product of all three cuts together: GW-band ISCO classification AND survey-window orbital period AND LS recovery probability. Starting from ~18,374 TNG mergers and applying those cuts leaves a handful of synthetic systems and zero real TNG100-1 systems recoverable from Stripe 82.

## Three-repo portfolio context

I built this as one piece of a three-repo multi-messenger SMBHB portfolio. The repos are meant to be read together.

| Repo | What it does |
|---|---|
| [smbhb-inspiral](https://github.com/WizardEternal/smbhb-inspiral) | Individual-system GW inspiral physics, sensitivity curves, EM detectability lookup |
| **tng-smbhb-population** (this repo) | TNG merger population, GW band classification, EM detectability, gap plot |
| [Stripe82-SMBHB-search](https://github.com/WizardEternal/Stripe82-SMBHB-search) | Periodic-quasar search in real SDSS Stripe 82 data (DRW + Lomb-Scargle) |

The arc is: IllustrisTNG progenitor population (this repo) feeds the GW inspiral physics ([smbhb-inspiral](https://github.com/WizardEternal/smbhb-inspiral)), which feeds the EM detectability gap (this repo, gap plot), which motivates the real-data EM search ([Stripe82-SMBHB-search](https://github.com/WizardEternal/Stripe82-SMBHB-search)).

## The gap plot

![Multi-messenger detectability funnel: ~5000 TNG merger proxies drop through quality cuts, GW band classification, survey window, and Lomb-Scargle recovery to a handful of recoverable candidates. Synthetic seed=42 canonical figure.](outputs/synthetic/gap_plot_stripe82.png)

The synthetic (seed=42) run starts with 5,000 mergers; after quality cuts, GW band classification, and the Stripe 82 window (200-1100 days), 9 systems remain. Applying Lin+2026 recovery fractions (45% sinusoidal, 9% sawtooth) gives ~4.1 sinusoidal-recoverable and ~0.8 sawtooth-recoverable. The real TNG100-1 run gives 0 at the Stripe 82 window step, for structural reasons explained in the Caveats section below. Regenerate with `python scripts/06_generate_plots.py --synthetic`.

## Caveats, read this first

Worth reading before trusting any number in the results table.

**Softening-scale mergers, not physical coalescences.** TNG BH mergers trigger when two BH particles come within the gravitational softening length (~0.7 kpc for TNG100-1). That's a numerical event, not a physical parsec-scale coalescence where GW emission becomes efficient. The catalog contains progenitor pairs that might become GW sources after an uncertain delay, not confirmed GW sources.

**The final parsec problem.** The delay from a TNG merger event to the GW-emitting phase is 100 Myr to several Gyr, and deeply uncertain. No confirmed mechanism efficiently shrinks all SMBHBs below ~1 pc across all mass ratios and environments. This repo makes no claim about that process.

**All GW band classifications are conditional.** Every classification in `gw_classification.py` answers: "IF this pair coalesces, which band carries its ISCO signal?" It doesn't assert coalescence will occur on any particular timescale.

**Seed-mass exclusion.** Mergers with M_tot < 1.2e6 M_sun are excluded as numerical artifacts of TNG's BH seeding prescription, not physically motivated mergers.

**EM detectability evaluated at f_ISCO, not time-averaged over inspiral.** Real inspirals spend most of their lifetime at longer orbital periods than the ISCO period. Evaluating at f_ISCO overstates the recoverable fraction. Time-weighted classification is deferred to Phase 3.

**Lin+2026 recovery fractions are population-averaged.** The 45% (sinusoidal) and 9% (sawtooth) numbers from Lin, Charisi and Haiman 2026 ([ApJ 997, 316](https://doi.org/10.3847/1538-4357/ae29a7)) are aggregate rates over a PTF-like survey population. Per-system rates depend on DRW noise parameters, signal amplitude, cadence, and period. Applying these fractions to individual TNG systems is an approximation.

**Synthetic vs real data.** The canonical gap plot and headline funnel numbers come from synthetic seed=42 data (5,000 mergers, log-uniform M_tot in [2e5, 2e10] M_sun, q in [0.1, 1]). The real TNG100-1 run (18,374 mergers) was corrected in Phase 2b: `bh_details.py` reads `blackhole_details.hdf5` (5.48 GB) and recovers the true primary BH mass for 100% of mergers, replacing the broken `mass_out` equal-mass proxy. Post-correction q distribution has median 0.113, 5/95 pct = 0.001/0.920, and M_tot range 2.36e6 to 1.24e10 M_sun. The real-data Stripe 82 = 0 result is a structural TNG100-1 mass-ceiling finding, not a bug: the heaviest system in the box (1.24e10 M_sun) gives P_orb_obs ~65 days at ISCO and z~0, below the 200-day Stripe 82 floor. The window requires M_tot >= 3.8e10 M_sun, which TNG100-1's (106.5 cMpc)^3 volume simply doesn't contain. For ultra-massive systems, use TNG300-1, or evaluate P_obs during hardening/inspiral rather than at ISCO.

## Why I built this

I wanted something concrete on the multi-messenger side before applying to PhD groups working on optical follow-up of compact-object mergers. My background is X-ray timing of BH X-ray binaries, and I'd been reading the GW/SMBHB literature for a while without having produced anything in that space myself. TNG gave me a public catalog I could actually work through: parse the HDF5, derive physical properties, classify GW bands, apply EM recovery fractions, produce the gap plot. The pipeline is kept simple so the physics assumptions are easy to see. I also wanted to produce the gap plot honestly, which means being upfront that TNG's "mergers" aren't physical coalescences and that the real-data run hits a hard mass ceiling.

## What's in here

```
tng-smbhb-population/
  README.md
  pyproject.toml
  src/tng_smbhb/
    catalog.py            load TNG blackhole_mergers.hdf5
    bh_details.py         Phase 2b: mine blackhole_details.hdf5 for true primary mass
    population.py         derive catalog properties (M_tot, q, z)
    gw_classification.py  f_ISCO band labels (PTA / LISA / gap)
    em_detectability.py   Lin+2026 recovery fractions per survey
    plotting.py           gap plot variants (Stripe82, LSST, dual-panel)
  scripts/                01 download, 02 parse, 03 fix masses, 04 classify GW, 05 classify EM, 06 plots
  data/                   place blackhole_mergers.hdf5 and blackhole_details.hdf5 here
  data/processed/         catalog_corrected.csv (committed, lets readers skip the pipeline)
  outputs/synthetic/      canonical seed=42 gap plot + CSVs
  outputs/real_corrected/ Phase 2b corrected real-data outputs
  tests/                  85 pytest tests
```

## How to run

```bash
pip install -e ".[dev]"
# Drop data/blackhole_mergers.hdf5 into data/ (download from tng-project.org, free account)
# For Phase 2b true masses also drop blackhole_details.hdf5 (5.48 GB)
python scripts/01_download_mergers.py
python scripts/02_parse_catalog.py
python scripts/03_fix_bh_masses.py    # Phase 2b, needs blackhole_details.hdf5
python scripts/04_classify_gw_bands.py
python scripts/05_classify_em_detectability.py
python scripts/06_generate_plots.py             # real data
python scripts/06_generate_plots.py --synthetic # canonical seed=42 run
```

Or use the Python API directly:

```python
from tng_smbhb.catalog import load_tng_hdf5
from tng_smbhb.population import derive_population
from tng_smbhb.gw_classification import classify_bands
from tng_smbhb.em_detectability import classify_em_detectability
from tng_smbhb.plotting import make_gap_plot

cat = load_tng_hdf5("data/blackhole_mergers.hdf5")
pop = derive_population(cat)
gwc = classify_bands(pop)
emc = classify_em_detectability(pop, gwc.f_isco_hz)
make_gap_plot(pop, gwc, emc, survey="stripe82", outpath="outputs/gap_plot.png")
```

If you don't want to run the 5.48 GB pipeline, `data/processed/catalog_corrected.csv` and everything in `outputs/` are committed to the repo so you can open the figures and CSVs directly.

## Results: headline numbers

Synthetic is the main run. The real-data column is there to show the pipeline works and to surface the mass-ceiling issue I ran into.

| Stage | Synthetic (seed=42, canonical) | Real TNG100-1 (corrected, Phase 2b) |
|---|---|---|
| All mergers | 5,000 | 18,374 |
| Quality cut (M_tot > 1.2e6 M_sun) | 4,085 (81.7%) | 18,374 (100%) |
| PTA-band f_ISCO (observer frame) | 1 (0.0%) | 0 (0.0%) |
| LISA-band f_ISCO (observer frame) | 1,377 (33.7%) | 5,885 (32.0%) |
| Gap-band f_ISCO (observer frame) | 2,698 (66.0%) | 12,489 (68.0%) |
| Stripe 82 window (200-1100 d) | 9 (0.22% of quality) | 0 (0%) |
| Sinusoidal-recoverable (x0.45) | ~4.1 | 0 |
| Sawtooth-recoverable (x0.09) | ~0.8 | 0 |

*Synthetic figures use seed=42, 5,000 mergers, log-uniform M_tot in [2e5, 2e10] M_sun, q in [0.1, 1]. Real-data figures use TNG100-1 `blackhole_mergers.hdf5` plus `blackhole_details.hdf5` with true primary masses recovered by `bh_details.py` (Phase 2b; q median 0.113). GW band counts use observer-frame f_ISCO = f_ISCO_source / (1 + z). Regenerate: `PYTHONPATH=src python scripts/06_generate_plots.py [--synthetic]`.*

The real-data Stripe 82 = 0 is structural: TNG100-1's heaviest merger is 1.24e10 M_sun, which gives P_orb_obs ~65 days at ISCO and z~0. That's below the 200-day Stripe 82 floor. Hitting the window requires M_tot >= 3.8e10 M_sun, and TNG100-1's box volume doesn't produce systems that massive. Synthetic data does produce Stripe 82 hits because the log-uniform M_tot upper bound of 2e10 M_sun is marginally above the structural threshold.

## Stuff that went sideways (and how I noticed)

1. **The `mass_out` field in `blackhole_mergers.hdf5` is broken.** For most mergers it equals `mass_in`, effectively forcing q = 1 everywhere. My first real-data run gave 9,976 LISA-band systems and looked plausible until I checked the q distribution and saw every entry was 1.0000. The fix (Phase 2b) was to mine `blackhole_details.hdf5` (5.48 GB, ~195M records) for the last pre-merger mass record of each surviving BH, cross-referenced against the `/unique` index for O(log N) lookup. 100% resolution rate. Post-fix q distribution: median 0.113, 5/95 pct = 0.001/0.920.

2. **Observer-frame vs source-frame f_ISCO.** My first GW classification pass used f_ISCO_source, which inflated the LISA count. Correcting to observer-frame f_ISCO = f_ISCO_source / (1 + z) dropped the LISA count from 1,593 to 1,377 in the synthetic run. Not a huge difference, but the observer frame is what actually matters for whether a detector on Earth can see it.

3. **Structural Stripe82=0 looks like a bug, isn't.** After Phase 2b the real-data Stripe 82 count stayed at 0 and I spent a while looking for a downstream error in the period calculation or survey-window logic. It wasn't a bug. TNG100-1's heaviest merger is 1.24e10 M_sun, and at ISCO with z~0 that gives P_orb_obs ~65 days. The 200-day floor cuts everything. This is a scientific finding about TNG100-1's mass ceiling, not a code problem. Synthetic data does produce Stripe 82 hits because the log-uniform upper bound at 2e10 M_sun is just above the structural threshold.

## Limitations

1. **Single-snapshot catalog.** No lightcone reconstruction. Everything is evaluated at the TNG merger snapshot redshift, which is a rough stand-in for the actual observational redshift of a GW or EM event.

2. **f_ISCO evaluated at coalescence, not time-weighted over inspiral.** Systems spend most of their inspiral lifetime at orbital periods much longer than the ISCO period. Evaluating detectability at ISCO overstates the fraction of the inspiral phase that's accessible to a survey. Time-weighted classification is a Phase 3 item.

3. **No BH spin.** TNG's sink-particle BH model doesn't evolve spin, and `smbhb-inspiral` uses the Schwarzschild ISCO in the spin-zero limit. There's no honest way to propagate spin through this pipeline, so I don't.

4. **Mass function is cut off at the TNG100-1 box-volume ceiling (~1.24e10 M_sun).** Cosmologically rare ultra-massive systems are absent. This is the root cause of the structural Stripe 82 = 0 result. TNG300-1 covers a larger volume and would produce higher-mass mergers.

5. **No host-galaxy matching.** The EM detectability cut is survey-window period plus LS recovery fraction only. It doesn't check whether the host galaxy is in the Stripe 82 footprint, above the survey magnitude limit, or at a redshift where the signal is detectable. A real yield estimate needs all of that.

6. **Lin+2026 fractions applied as aggregate constants, not per-system.** Phase 3 plan is to swap to per-system LS injection-recovery using MacLeod+2010 DRW scaling at Stripe 82 cadence.

7. **No lensing, no delay-time distribution integrated.** Delay-time is treated as a free parameter (or ignored), not folded into the yield estimate. Phase 3 LSST predictions will need a delay-time distribution.

8. **Error bars on the recoverable fractions are rough.** The "~4.1 sinusoidal, ~0.8 sawtooth" numbers carry Poisson uncertainty at best. The real uncertainty on ~1 sawtooth-recoverable is dominated by the Lin+2026 recovery-fraction uncertainty, which is itself population-averaged and wide.

## References

- Nelson et al. 2019, Computational Astrophysics and Cosmology 6, 2. *The IllustrisTNG Simulations: Public Data Release.* [doi:10.1186/s40668-019-0028-x](https://doi.org/10.1186/s40668-019-0028-x)
- Lin, Charisi & Haiman 2026, ApJ 997, 316. *Lomb-Scargle Periodogram Struggles with Non-sinusoidal Supermassive Black Hole Binary Signatures in Quasar Lightcurves.* [arXiv:2505.14778](https://arxiv.org/abs/2505.14778)
- Sesana et al. 2004, ApJ 611, 623. *Low-Frequency Gravitational Radiation from Coalescing Massive Black Hole Binaries in Hierarchical Cosmologies.* [doi:10.1086/422185](https://doi.org/10.1086/422185)
- Rajagopal & Romani 1995, ApJ 446, 543. *Ultra-Low-Frequency Gravitational Radiation from Massive Black Hole Binaries.* [doi:10.1086/175813](https://doi.org/10.1086/175813)
- Peters 1964, Phys. Rev. 136, B1224. *Gravitational Radiation and the Motion of Two Point Masses.* [doi:10.1103/PhysRev.136.B1224](https://doi.org/10.1103/PhysRev.136.B1224)
- Charisi et al. 2016, MNRAS 463, 2145. *A population of short-period variable quasars from PTF as supermassive black hole binary candidates.* [doi:10.1093/mnras/stw1838](https://doi.org/10.1093/mnras/stw1838)
- Graham et al. 2015, MNRAS 453, 1562. *A systematic search for close supermassive black hole binaries in the Catalina Real-time Transient Survey.* [doi:10.1093/mnras/stv1726](https://doi.org/10.1093/mnras/stv1726)
- Haiman, Kocsis & Menou 2009, ApJ 700, 1952. *The Population of Viscosity- and Gravitational Wave-Driven Supermassive Black Hole Binaries Among Luminous Active Galactic Nuclei.* [doi:10.1088/0004-637X/700/2/1952](https://doi.org/10.1088/0004-637X/700/2/1952)
- MacLeod et al. 2010, ApJ 721, 1014. *Modeling the Time Variability of SDSS Stripe 82 Quasars as a Damped Random Walk.* [doi:10.1088/0004-637X/721/2/1014](https://doi.org/10.1088/0004-637X/721/2/1014)

## Author

Karan Akbari, MSc Astrophysics, St. Xavier's College Mumbai. Background is X-ray timing and spectral analysis of black hole X-ray binaries (GRS 1915+105, 4U 1630-47) with Dr. Sudip Bhattacharyya at TIFR. Built this as part of a three-repo SMBHB multi-messenger portfolio while bridging into GW/compact-object follow-up work.

## License

MIT. See [LICENSE](LICENSE).
