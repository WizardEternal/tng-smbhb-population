"""Vectorized EM survey recoverability classification for TNG SMBHB populations.

Applies the Lin, Charisi & Haiman 2026 recovery fractions (vendored from
_vendored_em_detectability.py) to every system in a TNGPopulation, returning
per-catalog numpy arrays and aggregate counts suitable for the gap plot.

Limitation
----------
Rest-frame orbital period at ISCO: P_rest = 2 / f_isco_hz (seconds).
Observer-frame: P_obs = P_rest * (1 + z).  Windows and recovery fractions
come from the vendored Lin+2026 tables.

A real SMBHB spends only a vanishing fraction of its inspiral near ISCO.
Classifying "EM-detectability at ISCO" thus *overestimates* the rate at which
the observed period happens to sit inside the survey window.  This is the
Phase-1b definition per EXECUTION_PLAN.md §5.3; a proper time-weighted
classification is Phase 3 material (paper pathway).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .population import TNGPopulation
from ._vendored_em_detectability import (
    RECOVERY_FRACTIONS,
    SURVEY_WINDOWS,
)

__all__ = [
    "RECOVERY_FRACTIONS",
    "SURVEY_WINDOWS",
    "SECONDS_PER_DAY",
    "EMClassification",
    "classify_em_detectability",
]

SECONDS_PER_DAY: float = 86_400.0


@dataclass(frozen=True)
class EMClassification:
    """EM survey recoverability classification for a full TNGPopulation.

    All array fields have shape ``(N,)`` where N is the catalog size.

    Parameters
    ----------
    population : TNGPopulation
        The parent population from which this classification was derived.
    f_isco_hz : npt.NDArray[np.float64]
        Per-system ISCO GW frequency [Hz], shape (N,).  Passed in from the
        caller (typically GWClassification.f_isco_hz).
    p_orb_isco_rest_s : npt.NDArray[np.float64]
        Rest-frame orbital period at ISCO [s].  ``= 2 / f_isco_hz``.
    p_orb_isco_rest_days : npt.NDArray[np.float64]
        Rest-frame orbital period at ISCO [days].
    p_orb_isco_obs_days : npt.NDArray[np.float64]
        Observer-frame orbital period at ISCO [days].  ``= rest_days * (1+z)``.
    in_stripe82 : npt.NDArray[np.bool_]
        True where p_orb_isco_obs_days falls in the Stripe 82 window
        [200, 1100] days.
    in_ptf : npt.NDArray[np.bool_]
        True where p_orb_isco_obs_days falls in the PTF window [100, 600] days.
    in_lsst : npt.NDArray[np.bool_]
        True where p_orb_isco_obs_days falls in the LSST window [100, 1200] days.
    expected_sin_stripe82 : npt.NDArray[np.float64]
        Per-system sinusoidal recovery weight for Stripe 82 (0.45 or 0.0).
    expected_saw_stripe82 : npt.NDArray[np.float64]
        Per-system sawtooth recovery weight for Stripe 82 (0.09 or 0.0).
    expected_sin_lsst : npt.NDArray[np.float64]
        Per-system sinusoidal recovery weight for LSST (0.23 or 0.0).
    expected_saw_lsst : npt.NDArray[np.float64]
        Per-system sawtooth recovery weight for LSST (0.01 or 0.0).
    """

    population: TNGPopulation
    f_isco_hz: npt.NDArray[np.float64]
    p_orb_isco_rest_s: npt.NDArray[np.float64]
    p_orb_isco_rest_days: npt.NDArray[np.float64]
    p_orb_isco_obs_days: npt.NDArray[np.float64]
    in_stripe82: npt.NDArray[np.bool_]
    in_ptf: npt.NDArray[np.bool_]
    in_lsst: npt.NDArray[np.bool_]
    # Recovery expected counts (float, fraction-weighted):
    expected_sin_stripe82: npt.NDArray[np.float64]   # 0 or 0.45 per system
    expected_saw_stripe82: npt.NDArray[np.float64]   # 0 or 0.09 per system
    expected_sin_lsst: npt.NDArray[np.float64]       # 0 or 0.23 per system
    expected_saw_lsst: npt.NDArray[np.float64]       # 0 or 0.01 per system

    # ------------------------------------------------------------------
    # Aggregate counts for the gap plot
    # ------------------------------------------------------------------

    @property
    def n_stripe82_window(self) -> int:
        """Number of systems with P_obs inside the Stripe 82 window."""
        return int(np.sum(self.in_stripe82))

    @property
    def n_lsst_window(self) -> int:
        """Number of systems with P_obs inside the LSST window."""
        return int(np.sum(self.in_lsst))

    @property
    def expected_n_sin_stripe82(self) -> float:
        """Expected sinusoidal recovery count for Stripe 82 (sum of per-system weights)."""
        return float(np.sum(self.expected_sin_stripe82))

    @property
    def expected_n_saw_stripe82(self) -> float:
        """Expected sawtooth recovery count for Stripe 82 (sum of per-system weights)."""
        return float(np.sum(self.expected_saw_stripe82))

    @property
    def expected_n_sin_lsst(self) -> float:
        """Expected sinusoidal recovery count for LSST (sum of per-system weights)."""
        return float(np.sum(self.expected_sin_lsst))

    @property
    def expected_n_saw_lsst(self) -> float:
        """Expected sawtooth recovery count for LSST (sum of per-system weights)."""
        return float(np.sum(self.expected_saw_lsst))


def classify_em_detectability(
    pop: TNGPopulation,
    f_isco_hz: npt.NDArray[np.float64],
) -> EMClassification:
    """Vectorized EM recoverability classification for all systems.

    Parameters
    ----------
    pop : TNGPopulation
        The TNG merger population to classify.
    f_isco_hz : array-like, shape (N,)
        Per-system f_ISCO [Hz].  Caller typically passes
        ``GWClassification.f_isco_hz``.

    Returns
    -------
    EMClassification
        Frozen dataclass with per-system boolean arrays, fraction-weighted
        expected-count arrays, and aggregate count properties.

    Notes
    -----
    Rest-frame orbital period at ISCO: P_rest = 2 / f_isco_hz (seconds).
    Observer-frame: P_obs = P_rest * (1 + z).  Windows and recovery
    fractions come from the vendored Lin+2026 tables.

    Limitation: a real SMBHB spends only a vanishing fraction of its
    inspiral near ISCO.  Classifying "EM-detectability at ISCO" thus
    *overestimates* the rate at which the observed period happens to sit
    inside the survey window.  This is the Phase-1b definition per
    EXECUTION_PLAN.md §5.3; a proper time-weighted classification is
    Phase 3 material (paper pathway).
    """
    f_isco: npt.NDArray[np.float64] = np.asarray(f_isco_hz, dtype=np.float64)
    z: npt.NDArray[np.float64] = np.asarray(pop.catalog.redshift, dtype=np.float64)

    # Rest-frame orbital period at ISCO: P = 2 / f_ISCO  (orbital period, seconds)
    p_rest_s: npt.NDArray[np.float64] = 2.0 / f_isco
    p_rest_days: npt.NDArray[np.float64] = p_rest_s / SECONDS_PER_DAY

    # Observer-frame period
    p_obs_days: npt.NDArray[np.float64] = p_rest_days * (1.0 + z)

    # Survey window flags — inclusive on both edges
    s82_min: float = float(SURVEY_WINDOWS["stripe82"]["P_min_days"])
    s82_max: float = float(SURVEY_WINDOWS["stripe82"]["P_max_days"])
    ptf_min: float = float(SURVEY_WINDOWS["ptf"]["P_min_days"])
    ptf_max: float = float(SURVEY_WINDOWS["ptf"]["P_max_days"])
    lsst_min: float = float(SURVEY_WINDOWS["lsst"]["P_min_days"])
    lsst_max: float = float(SURVEY_WINDOWS["lsst"]["P_max_days"])

    in_s82: npt.NDArray[np.bool_] = (p_obs_days >= s82_min) & (p_obs_days <= s82_max)
    in_ptf: npt.NDArray[np.bool_] = (p_obs_days >= ptf_min) & (p_obs_days <= ptf_max)
    in_lsst: npt.NDArray[np.bool_] = (p_obs_days >= lsst_min) & (p_obs_days <= lsst_max)

    # Recovery fractions from Lin, Charisi & Haiman 2026
    # Stripe 82 uses ptf_like cadence; LSST uses lsst_like cadence
    frac_sin_s82: float = RECOVERY_FRACTIONS["sinusoidal"]["ptf_like"]   # 0.45
    frac_saw_s82: float = RECOVERY_FRACTIONS["sawtooth"]["ptf_like"]     # 0.09
    frac_sin_lsst: float = RECOVERY_FRACTIONS["sinusoidal"]["lsst_like"] # 0.23
    frac_saw_lsst: float = RECOVERY_FRACTIONS["sawtooth"]["lsst_like"]   # 0.01

    # Per-system expected counts (fraction * 1 if in window, else 0)
    exp_sin_s82: npt.NDArray[np.float64] = frac_sin_s82 * in_s82.astype(np.float64)
    exp_saw_s82: npt.NDArray[np.float64] = frac_saw_s82 * in_s82.astype(np.float64)
    exp_sin_lsst: npt.NDArray[np.float64] = frac_sin_lsst * in_lsst.astype(np.float64)
    exp_saw_lsst: npt.NDArray[np.float64] = frac_saw_lsst * in_lsst.astype(np.float64)

    return EMClassification(
        population=pop,
        f_isco_hz=f_isco,
        p_orb_isco_rest_s=p_rest_s,
        p_orb_isco_rest_days=p_rest_days,
        p_orb_isco_obs_days=p_obs_days,
        in_stripe82=in_s82,
        in_ptf=in_ptf,
        in_lsst=in_lsst,
        expected_sin_stripe82=exp_sin_s82,
        expected_saw_stripe82=exp_saw_s82,
        expected_sin_lsst=exp_sin_lsst,
        expected_saw_lsst=exp_saw_lsst,
    )
