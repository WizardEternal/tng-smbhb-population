"""IllustrisTNG black-hole merger catalog loader with physical unit conversion.

This module reads the TNG ``blackhole_mergers.hdf5`` (or equivalent) HDF5 file,
converts masses from TNG internal units (10^10 M_sun / h) to solar masses,
and returns a validated, immutable :class:`TNGMergerCatalog` dataclass.

HDF5 schema (IllustrisTNG public release, 2016-09-28, git e659973)
------------------------------------------------------------------
Root-level datasets::

    /time          (N,)  float32   Scale factor a of the merger snapshot.
    /mass_in       (N,)  float32   Infalling ("in") BH mass [10^10 M_sun / h].
    /mass_out      (N,)  float64   See CRITICAL CAVEAT below.
    /id_in         (N,)  uint64    Particle ID of consumed BH.
    /id_out        (N,)  uint64    Particle ID of surviving BH.
    /snapshot      (N,)  int64     Snapshot index of the merger.
    /Header        group           Metadata attributes.
    /details       group           Per-BH context (cs, mass, mdot, rho, time).
    /tree          group           Merger-tree ancestry.

CRITICAL TNG DATA CAVEAT — ``mass_out`` is unusable
---------------------------------------------------
The ``/Header`` group's ``description`` attribute of the public TNG100-1
``blackhole_mergers.hdf5`` file contains the following warning verbatim:

    "NOTE: the mass of the 'out' BH is incorrect in this data. The values
    given correspond to the total cell (dynamical) mass, instead of the BH
    mass itself."

I (Karan) verified this on the 18,374-merger TNG100-1 catalog (SHA check):
``/details/mass[:, 1]`` agrees with ``/mass_out`` to within float32 precision
for all 17,736 rows where it is non-zero (the remaining 638 rows have
``col1 = 0``), so the ``/details`` subgroup does not rescue the primary BH
mass — it replicates the same broken value.  There is **no usable record of
the pre-merger primary BH mass** anywhere in this file.

Mass assignment (Phase 1b equal-mass proxy)
-------------------------------------------
Given the bug above, Phase 1b treats every merger as an **equal-mass proxy**:

    m2 = mass_in         (the consumed/secondary BH; this field IS correct)
    m1 = mass_in         (proxy for the primary; a documented lower bound)

This yields a total mass ``M_tot = 2 * mass_in`` and mass ratio ``q = 1``.
Consequences for downstream analysis:

* Total mass is a lower bound (the real primary is ``>= mass_in`` in the
  majority of cases).
* Chirp mass ``M_c = (m1 m2)^(3/5) / (m1 + m2)^(1/5)`` is also a lower bound.
* ``f_ISCO ∝ 1 / M_tot`` is therefore an *upper* bound (overestimate), so the
  LISA-band count is biased high and the gap-region count biased low relative
  to reality with correct primary masses.
* Mass ratio distribution is pinned to ``q = 1`` (the TNG population actually
  has a distribution of ``q`` from ~0.01 to ~1, which will be recovered in
  Phase 2b via the TNG API).

The proper fix is implemented in Phase 2b as
``scripts/03_fix_bh_masses.py``, which reads the ``/mass`` field of the
TNG supplementary ``blackhole_details.hdf5`` file (the per-timestep BH mass
is unmodified from the raw simulation there) and writes
``data/processed/catalog_corrected.csv``.  See
:mod:`tng_smbhb.bh_details` for the lookup API.  (Original Phase 2b plan
was TNG API ``SubhaloBHMass`` queries per EXECUTION_PLAN.md §7.2 — abandoned
in favor of the details-file path, which is O(10^4) faster and
rate-limit-free.)

For the portfolio narrative, the **synthetic-data Gate 2 figures**
(seed=42, 5000 mergers, drawn from an ``M_tot in [2e5, 2e10] M_sun`` log-uniform
with ``q in [0.1, 1]`` uniform) remain the canonical multi-messenger gap-plot
deliverable.  The real-data run here is a supplementary sanity check that
the loader, quality cut, GW-band classifier, and EM-detectability stages
complete cleanly on the live HDF5 file.

Field provenance
----------------
``time``
    Scale factor *a* of the merger snapshot (dimensionless).
``mass_in``
    Consumed (secondary) BH mass in TNG internal units: 10^10 M_sun / h.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import h5py as h5py_typing  # noqa: F401  (type-checking only)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: TNG internal mass unit expressed in M_sun per unit per h.
#: One TNG mass unit = 10^10 M_sun / h.
_TNG_MASS_UNIT_OVER_H: float = 1.0e10  # M_sun / h


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TNGMergerCatalog:
    """Parsed TNG black-hole merger catalog, in physical units (M_sun).

    Attributes
    ----------
    m1_msun : npt.NDArray[np.float64]
        Primary (heavier) BH mass at merger, shape ``(N,)``  [M_sun].
        Guaranteed ``m1_msun >= m2_msun`` element-wise.
    m2_msun : npt.NDArray[np.float64]
        Secondary (lighter) BH mass at merger, shape ``(N,)``  [M_sun].
    scale_factor : npt.NDArray[np.float64]
        Cosmic scale factor *a* at the merger event, shape ``(N,)``.
        Ranges over ``(0, 1]``; ``a = 1`` corresponds to z = 0.
    redshift : npt.NDArray[np.float64]
        Redshift ``z = 1/a - 1``, shape ``(N,)``.
    simulation : str
        Short name of the originating TNG run, e.g. ``"TNG100-1"``.
    hubble_h : float
        Dimensionless Hubble parameter used for unit conversion (e.g. 0.6774
        for TNG / Planck15).
    n_mergers : int
        Number of merger events in the catalog (equal to ``len(m1_msun)``).
    """

    m1_msun: npt.NDArray[np.float64]
    m2_msun: npt.NDArray[np.float64]
    scale_factor: npt.NDArray[np.float64]
    redshift: npt.NDArray[np.float64]
    simulation: str
    hubble_h: float
    n_mergers: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_tng_mass(raw: npt.NDArray[np.float64], hubble_h: float) -> npt.NDArray[np.float64]:
    """Convert TNG internal mass units to solar masses.

    Parameters
    ----------
    raw : npt.NDArray[np.float64]
        Mass array in TNG internal units (10^10 M_sun / h).
    hubble_h : float
        Dimensionless Hubble parameter *h*.

    Returns
    -------
    npt.NDArray[np.float64]
        Mass array in M_sun.
    """
    return raw * (_TNG_MASS_UNIT_OVER_H / hubble_h)


def _read_datasets(
    hdf5_path: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Open *hdf5_path* and return ``(time, mass_in)`` arrays.

    Reads the two physically-meaningful root-level datasets from the real
    TNG100-1 ``blackhole_mergers.hdf5`` schema (dataset names are lowercase).
    ``mass_out`` is *not* read because the TNG Header warns it is the
    dynamical cell mass, not the BH mass (see module docstring).

    Parameters
    ----------
    hdf5_path : Path
        Resolved path to the HDF5 file.

    Returns
    -------
    tuple of npt.NDArray[np.float64]
        ``(time, mass_in)`` cast to ``np.float64``.

    Raises
    ------
    ValueError
        If the expected datasets are not found.  Tries lowercase
        (real TNG schema) first, then falls back to capitalized names for
        historical-format or test-fixture compatibility.
    """
    import h5py  # noqa: PLC0415

    # Real TNG schema uses lowercase; retain capitalized fallback for
    # test fixtures and any alternative catalog formats.
    _LOWER = ("time", "mass_in")
    _UPPER = ("Time", "BHMass_In")

    with h5py.File(hdf5_path, "r") as f:
        if all(k in f for k in _LOWER):
            return (
                np.asarray(f[_LOWER[0]], dtype=np.float64),
                np.asarray(f[_LOWER[1]], dtype=np.float64),
            )
        if all(k in f for k in _UPPER):
            return (
                np.asarray(f[_UPPER[0]], dtype=np.float64),
                np.asarray(f[_UPPER[1]], dtype=np.float64),
            )
        if "Mergers" in f:
            grp = f["Mergers"]
            if all(k in grp for k in _LOWER):
                return (
                    np.asarray(grp[_LOWER[0]], dtype=np.float64),
                    np.asarray(grp[_LOWER[1]], dtype=np.float64),
                )
            if all(k in grp for k in _UPPER):
                return (
                    np.asarray(grp[_UPPER[0]], dtype=np.float64),
                    np.asarray(grp[_UPPER[1]], dtype=np.float64),
                )

    raise ValueError(
        f"HDF5 file '{hdf5_path}' does not contain the required datasets. "
        f"Expected either ('time', 'mass_in') [real TNG schema] or "
        f"('Time', 'BHMass_In') [historical], at root level or under a "
        f"'Mergers' subgroup."
    )


def _build_catalog(
    time_raw: npt.NDArray[np.float64],
    mass_in_raw: npt.NDArray[np.float64],
    *,
    simulation: str,
    hubble_h: float,
) -> TNGMergerCatalog:
    """Convert raw TNG arrays to a :class:`TNGMergerCatalog` (equal-mass proxy).

    See the module docstring for the Phase 1b rationale: the primary BH mass
    is not recoverable from the TNG HDF5 file due to a documented TNG data
    bug (``mass_out`` is the dynamical cell mass), so ``m1 = m2 = mass_in``.

    Parameters
    ----------
    time_raw : npt.NDArray[np.float64]
        Scale-factor array (already dimensionless).
    mass_in_raw : npt.NDArray[np.float64]
        Secondary BH mass in TNG internal units.
    simulation : str
        Short simulation name.
    hubble_h : float
        Dimensionless Hubble parameter.

    Returns
    -------
    TNGMergerCatalog
        Validated, unit-converted catalog with ``m1 = m2 = mass_in``
        (equal-mass proxy; see module docstring).
    """
    # Unit-convert the one mass we trust.
    m_sec = _convert_tng_mass(mass_in_raw, hubble_h)

    # Valid-row mask: finite, positive mass and scale factor.
    valid_mask = (
        np.isfinite(m_sec)
        & (m_sec > 0.0)
        & np.isfinite(time_raw)
        & (time_raw > 0.0)
    )

    n_corrupt = int(np.sum(~valid_mask))
    if n_corrupt > 0:
        warnings.warn(
            f"Skipping {n_corrupt} corrupt row(s) in TNG catalog "
            f"(non-positive or NaN mass_in / time).",
            stacklevel=3,
        )

    m_sec = m_sec[valid_mask]
    a = time_raw[valid_mask]

    # Equal-mass proxy: m1 = m2 = mass_in.  See module docstring.
    m1 = m_sec.copy()
    m2 = m_sec.copy()

    redshift = 1.0 / a - 1.0

    return TNGMergerCatalog(
        m1_msun=m1.astype(np.float64),
        m2_msun=m2.astype(np.float64),
        scale_factor=a.astype(np.float64),
        redshift=redshift.astype(np.float64),
        simulation=simulation,
        hubble_h=hubble_h,
        n_mergers=int(len(m1)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_tng_hdf5(
    path: str | Path,
    *,
    simulation: str = "TNG100-1",
    hubble_h: float = 0.6774,
) -> TNGMergerCatalog:
    """Load and unit-convert a TNG black-hole merger HDF5 file.

    Reads ``time`` and ``mass_in`` from the real TNG HDF5 schema (or the
    capitalized historical names, for test fixtures), converts the
    secondary mass from TNG internal units (10^10 M_sun / h) to solar
    masses, drops corrupt rows, and returns an immutable
    :class:`TNGMergerCatalog`.

    .. warning::

       The primary BH mass is set equal to the secondary (``m1 = m2 =
       mass_in``) because the TNG public HDF5 file's ``mass_out`` field is
       known to contain the dynamical cell mass rather than the post-merger
       BH mass (see module docstring).  This is a Phase 1b workaround;
       Phase 2b will recover correct primary masses via the TNG API.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the HDF5 file (e.g. ``blackhole_mergers.hdf5``).
    simulation : str, optional
        Human-readable name of the TNG run.  Default ``"TNG100-1"``.
    hubble_h : float, optional
        Dimensionless Hubble parameter *h* used in the TNG internal mass
        unit (10^10 M_sun / h).  Default ``0.6774`` (Planck 2015 / TNG).

    Returns
    -------
    TNGMergerCatalog
        Parsed catalog with masses in M_sun and redshifts computed from the
        snapshot scale factors.  ``m1_msun == m2_msun`` element-wise by
        construction (equal-mass proxy).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on the filesystem.
    ValueError
        If the HDF5 file lacks the required datasets, or if *hubble_h* is
        non-positive.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TNG HDF5 file not found: '{path}'.")
    if hubble_h <= 0.0:
        raise ValueError(f"hubble_h must be positive; got {hubble_h}.")

    time_raw, mass_in_raw = _read_datasets(path)

    return _build_catalog(
        time_raw,
        mass_in_raw,
        simulation=simulation,
        hubble_h=hubble_h,
    )


def catalog_from_arrays(
    m1_msun: npt.ArrayLike,
    m2_msun: npt.ArrayLike,
    scale_factor: npt.ArrayLike,
    *,
    simulation: str,
    hubble_h: float,
) -> TNGMergerCatalog:
    """Construct a :class:`TNGMergerCatalog` directly from arrays.

    Useful for synthetic catalogs, unit tests, and downstream analysis
    pipelines that generate merger populations programmatically rather than
    loading a TNG HDF5 file.  Unlike :func:`load_tng_hdf5`, this constructor
    accepts independent ``m1`` and ``m2`` (so synthetic populations *can*
    have ``q < 1``).

    Parameters
    ----------
    m1_msun : array_like
        Primary BH masses  [M_sun].  Need not satisfy ``m1 >= m2`` on entry;
        the function will swap component masses where needed.
    m2_msun : array_like
        Secondary BH masses  [M_sun].
    scale_factor : array_like
        Cosmic scale factor *a* at each merger event.  Must be positive.
    simulation : str
        Human-readable name for the originating simulation run.
    hubble_h : float
        Dimensionless Hubble parameter (stored as metadata; not used in any
        conversion because input masses are already in M_sun).

    Returns
    -------
    TNGMergerCatalog
        Immutable catalog with ``m1_msun >= m2_msun`` element-wise and
        ``redshift = 1 / scale_factor - 1``.

    Raises
    ------
    ValueError
        If array shapes do not match, if any mass is non-positive or NaN, or
        if any scale factor is non-positive.

    Notes
    -----
    All inputs are cast to ``np.float64`` before validation.
    """
    m1 = np.asarray(m1_msun, dtype=np.float64).ravel()
    m2 = np.asarray(m2_msun, dtype=np.float64).ravel()
    a = np.asarray(scale_factor, dtype=np.float64).ravel()

    if m1.shape != m2.shape:
        raise ValueError(
            f"m1_msun and m2_msun must have the same shape; "
            f"got {m1.shape} and {m2.shape}."
        )
    if m1.shape != a.shape:
        raise ValueError(
            f"scale_factor must have the same shape as m1_msun; "
            f"got {a.shape} vs {m1.shape}."
        )

    # Validate positivity
    if not np.all(np.isfinite(m1) & (m1 > 0.0)):
        raise ValueError("All m1_msun values must be finite and positive.")
    if not np.all(np.isfinite(m2) & (m2 > 0.0)):
        raise ValueError("All m2_msun values must be finite and positive.")
    if not np.all(np.isfinite(a) & (a > 0.0)):
        raise ValueError("All scale_factor values must be finite and positive.")

    # Enforce m1 >= m2
    swap = m2 > m1
    m1_out = np.where(swap, m2, m1)
    m2_out = np.where(swap, m1, m2)

    redshift = 1.0 / a - 1.0

    return TNGMergerCatalog(
        m1_msun=m1_out,
        m2_msun=m2_out,
        scale_factor=a,
        redshift=redshift,
        simulation=simulation,
        hubble_h=hubble_h,
        n_mergers=int(len(m1_out)),
    )
