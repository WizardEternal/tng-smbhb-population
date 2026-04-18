# Data Provenance

This file documents the origin and expected format of all data used by this project.

---

## `blackhole_mergers.hdf5` (TNG100-1)

- **Source:** IllustrisTNG public data release — https://www.tng-project.org
- **Access:** Downloaded manually with a registered tng-project.org account. The file is NOT committed to this repository.
- **Expected path:** `data/blackhole_mergers.hdf5`
- **Expected HDF5 datasets:**

  | Dataset | Description |
  |---|---|
  | `Time` | Scale factor $a$ at the moment of merger |
  | `BHMass_In` | Mass of the infalling (smaller) BH, in $10^{10}\,M_\odot / h$ |
  | `BHMass_Out` | Mass of the remaining (larger) BH after merger, in $10^{10}\,M_\odot / h$ |

- **Cosmology:** Planck 2015. Hubble parameter $h = 0.6774$ (used for all unit conversions).
- **Physical caveat:** TNG mergers are recorded when two BH particles fall within the gravitational softening length (~0.7 kpc comoving). This is NOT a physical coalescence. The delay from this numerical merger to the actual GW-emitting inspiral phase is estimated at 100 Myr to several Gyr and is deeply uncertain. All downstream counts must be understood in this context.

---

## Units as used in this repo

- **Masses:** converted to $M_\odot$ from TNG internal units via $M = M_\mathrm{TNG} \times 10^{10} / h$.
- **Redshift:** derived from scale factor as $z = 1/a - 1$.

---

## Lin+2026 locked numbers

Recovery fractions from Lin, Charisi & Haiman (2026), ApJ 997, 316
(DOI: 10.3847/1538-4357/ae29a7), Table 1.

| Survey tier | Sinusoidal | Sawtooth |
|---|---|---|
| PTF-like | 45% | 9% |
| Idealized | 24% | 1% |
| LSST-like | 23% | 1% |

Author quote (Lin et al. 2026): "Previous searches, including the one in M. Charisi et al. (2016), must have missed a significant fraction of periodic signals."

These numbers are treated as fixed inputs to `em_detectability.py` and must not be adjusted without updating the citation.

---

*Last updated: 2026-04-17*
