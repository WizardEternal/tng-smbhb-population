"""Population-level quantities derived from a TNGMergerCatalog.

Vectorized over full catalog arrays — no Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tng_smbhb.catalog import TNGMergerCatalog, catalog_from_arrays

__all__ = [
    "TNGPopulation",
    "derive_population",
    "filter_population",
]

_DEFAULT_MIN_TOTAL_MASS: float = 1.2e6  # M_sun


# ---------------------------------------------------------------------------
# Vectorized physics formulas (inline, array-native — no Python loops)
# ---------------------------------------------------------------------------


def _chirp_mass_vec(
    m1: npt.NDArray[np.float64],
    m2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Chirp mass in the same units as m1, m2.

    M_c = (m1 * m2)^0.6 / (m1 + m2)^0.2
    """
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def _eta_vec(
    m1: npt.NDArray[np.float64],
    m2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Symmetric mass ratio  eta = m1*m2 / (m1+m2)^2."""
    return m1 * m2 / (m1 + m2) ** 2


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TNGPopulation:
    """Population quantities derived from a TNGMergerCatalog."""

    catalog: TNGMergerCatalog
    chirp_mass_msun: npt.NDArray[np.float64]  # (N,)
    total_mass_msun: npt.NDArray[np.float64]  # (N,)
    mass_ratio_q: npt.NDArray[np.float64]     # (N,), q = m2/m1 in (0, 1]
    eta: npt.NDArray[np.float64]              # (N,), symmetric mass ratio
    passes_quality_cut: npt.NDArray[np.bool_] # (N,)
    quality_cut_min_total_mass_msun: float    # 1.2e6

    @property
    def n_total(self) -> int:
        """Total number of mergers in the underlying catalog."""
        return int(self.catalog.n_mergers)

    @property
    def n_passing(self) -> int:
        """Number of mergers that pass the quality cut."""
        return int(np.sum(self.passes_quality_cut))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive_population(
    catalog: TNGMergerCatalog,
    *,
    min_total_mass_msun: float = _DEFAULT_MIN_TOTAL_MASS,
) -> TNGPopulation:
    """Derive chirp mass, total mass, q, eta, and quality-cut mask.

    Quality cut: exclude systems with M_tot < min_total_mass_msun (seed-mass BHs),
    and any row with non-positive masses (shouldn't happen, but guard).

    Default min_total_mass_msun = 1.2e6 per EXECUTION_PLAN.md: TNG seed mass is
    ~8e5 M_sun/h ≈ 1.18e6 M_sun at h=0.6774; we round up to 1.2e6 to cleanly
    exclude seed-seed and near-seed-mass mergers.
    """
    m1: npt.NDArray[np.float64] = np.asarray(catalog.m1_msun, dtype=np.float64)
    m2: npt.NDArray[np.float64] = np.asarray(catalog.m2_msun, dtype=np.float64)

    total_mass: npt.NDArray[np.float64] = m1 + m2
    chirp_mass: npt.NDArray[np.float64] = _chirp_mass_vec(m1, m2)
    # Suppress divide-by-zero: rows with m1==0 are excluded by quality cut below.
    with np.errstate(divide="ignore", invalid="ignore"):
        q: npt.NDArray[np.float64] = m2 / m1
    eta: npt.NDArray[np.float64] = _eta_vec(m1, m2)

    passes: npt.NDArray[np.bool_] = (
        (total_mass > min_total_mass_msun) & (m1 > 0.0) & (m2 > 0.0)
    )

    return TNGPopulation(
        catalog=catalog,
        chirp_mass_msun=chirp_mass,
        total_mass_msun=total_mass,
        mass_ratio_q=q,
        eta=eta,
        passes_quality_cut=passes,
        quality_cut_min_total_mass_msun=min_total_mass_msun,
    )


def filter_population(pop: TNGPopulation) -> TNGPopulation:
    """Return a new TNGPopulation containing only rows where passes_quality_cut is True.

    The catalog inside is rebuilt from the filtered arrays.
    """
    mask: npt.NDArray[np.bool_] = pop.passes_quality_cut

    filtered_catalog = catalog_from_arrays(
        m1_msun=pop.catalog.m1_msun[mask],
        m2_msun=pop.catalog.m2_msun[mask],
        scale_factor=pop.catalog.scale_factor[mask],
        simulation=pop.catalog.simulation,
        hubble_h=pop.catalog.hubble_h,
    )

    return TNGPopulation(
        catalog=filtered_catalog,
        chirp_mass_msun=pop.chirp_mass_msun[mask],
        total_mass_msun=pop.total_mass_msun[mask],
        mass_ratio_q=pop.mass_ratio_q[mask],
        eta=pop.eta[mask],
        passes_quality_cut=pop.passes_quality_cut[mask],
        quality_cut_min_total_mass_msun=pop.quality_cut_min_total_mass_msun,
    )
