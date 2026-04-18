"""GW band classification for TNG SMBHB merger populations.

Frame convention (CRITICAL)
----------------------------
Gravitational-wave detectors (PTA arrays, LISA) measure signals in the
**observer (detector) frame**.  The ISCO frequency is computed in the
**source rest frame** from the binary's total mass:

    f_ISCO_source = c^3 / (6^(3/2) * pi * G * M_tot)

The observed frequency is redshifted:

    f_ISCO_observer = f_ISCO_source / (1 + z)

Band classification MUST use the observer-frame frequency because that is
what the detector actually measures.  Using the source-frame frequency
incorrectly assigns high-redshift systems to higher-frequency bands,
over-counting LISA-band systems and under-counting PTA-band and gap systems.

This module exposes BOTH fields:
  - ``f_isco_source_hz`` — source-frame frequency (diagnostic; also
    passed to ``em_detectability.py`` for the period calculation, which
    applies its own (1+z) factor internally).
  - ``f_isco_observer_hz`` — observer-frame frequency (used for band
    assignment).

Band edges (locked — do not change, see EXECUTION_PLAN.md L6/L7):

    PTA    : 1e-9 <= f_ISCO_observer <= 1e-7 Hz
    gap    : 1e-7 <  f_ISCO_observer <  1e-4 Hz   (pulsar-to-LISA gap)
    LISA   : 1e-4 <= f_ISCO_observer <= 1e-1 Hz
    neither: f_ISCO_observer < 1e-9 or f_ISCO_observer > 1e-1 Hz

Rationale: The 7e8 M_sun reference system used in smbhb-inspiral has
f_ISCO_source ≈ 6e-6 Hz, which falls in the "gap" band at z≈0 — this is
physically correct. Supermassive systems (> ~few × 1e9 M_sun) land in the
PTA band; lighter systems (< ~1e6 M_sun) land in the LISA band. Systems
with M_tot ~1e8 M_sun sit in the gap, which is a key pedagogical point of
the portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import pi

import numpy as np
import numpy.typing as npt

from tng_smbhb._vendored_constants import G, M_SUN, c
from tng_smbhb.population import TNGPopulation

__all__ = [
    "GWBand",
    "GWClassification",
    "PTA_F_MIN_HZ",
    "PTA_F_MAX_HZ",
    "LISA_F_MIN_HZ",
    "LISA_F_MAX_HZ",
    "compute_f_isco",
    "compute_f_isco_observer",
    "classify_bands",
]

# ---------------------------------------------------------------------------
# Band boundary constants (locked — do not change)
# ---------------------------------------------------------------------------

PTA_F_MIN_HZ: float = 1e-9
PTA_F_MAX_HZ: float = 1e-7
LISA_F_MIN_HZ: float = 1e-4
LISA_F_MAX_HZ: float = 1e-1


# ---------------------------------------------------------------------------
# Band enum
# ---------------------------------------------------------------------------


class GWBand(str, Enum):
    """GW frequency band labels."""

    PTA = "pta"
    LISA = "lisa"
    GAP = "gap"
    NEITHER = "neither"


# ---------------------------------------------------------------------------
# Vectorized f_ISCO
# ---------------------------------------------------------------------------


def compute_f_isco(
    total_mass_msun: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Vectorized ISCO GW frequency (source rest frame) for an array of total masses.

    For a Schwarzschild (non-spinning) black hole of total mass M the ISCO
    sits at r_ISCO = 6 G M / c^2.  Kepler's third law gives the orbital
    frequency; the GW frequency is twice that:

        f_ISCO_source = c^3 / (6^(3/2) * pi * G * M_tot_kg)

    This is the SOURCE-FRAME frequency.  For band classification use
    :func:`compute_f_isco_observer` which applies the (1+z) redshift factor.

    Parameters
    ----------
    total_mass_msun : npt.NDArray[np.float64]
        Total binary mass in solar masses, shape (N,).

    Returns
    -------
    npt.NDArray[np.float64]
        ISCO GW frequency in the source rest frame [Hz], shape (N,).

    Notes
    -----
    Uses G, c, M_SUN from _vendored_constants.  Pure NumPy — no Python loops.
    """
    m_tot_kg: npt.NDArray[np.float64] = np.asarray(
        total_mass_msun, dtype=np.float64
    ) * M_SUN
    return c**3 / (6.0**1.5 * pi * G * m_tot_kg)


def compute_f_isco_observer(
    total_mass_msun: npt.NDArray[np.float64],
    redshift: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Vectorized ISCO GW frequency in the observer (detector) frame.

    Applies the cosmological redshift factor to the source-frame ISCO frequency:

        f_ISCO_observer = f_ISCO_source / (1 + z)

    This is the frequency that PTA and LISA detectors actually measure.
    Band classification MUST use this quantity.

    Parameters
    ----------
    total_mass_msun : npt.NDArray[np.float64]
        Total binary mass in solar masses, shape (N,).
    redshift : npt.NDArray[np.float64]
        Cosmological redshift z for each system, shape (N,).

    Returns
    -------
    npt.NDArray[np.float64]
        ISCO GW frequency in the observer frame [Hz], shape (N,).
    """
    f_source = compute_f_isco(total_mass_msun)
    z: npt.NDArray[np.float64] = np.asarray(redshift, dtype=np.float64)
    return f_source / (1.0 + z)


# ---------------------------------------------------------------------------
# Classification dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GWClassification:
    """Per-system GW band classification, aligned 1-to-1 with pop.catalog.

    Attributes
    ----------
    population : TNGPopulation
        The source population (all systems, including non-quality-passing ones).
    f_isco_source_hz : npt.NDArray[np.float64]
        ISCO GW frequency in the source rest frame, shape (N,)  [Hz].
        Diagnostic field; do NOT use for band assignment.  Passed through
        to ``em_detectability.py``, which applies its own (1+z) factor for
        the period calculation.
    f_isco_observer_hz : npt.NDArray[np.float64]
        ISCO GW frequency in the observer (detector) frame, shape (N,)  [Hz].
        ``= f_isco_source_hz / (1 + z)``.  This is what PTA/LISA detectors
        measure, and is the quantity used for band assignment.
    f_isco_hz : npt.NDArray[np.float64]
        Alias for ``f_isco_source_hz`` for backward compatibility with
        ``em_detectability.py`` call sites.  Use ``f_isco_observer_hz``
        for any new band-classification logic.
    band : npt.NDArray[np.str_]
        Band label for each system, shape (N,).
        Values: "pta" / "lisa" / "gap" / "neither".
        Assigned using ``f_isco_observer_hz``.
    in_pta : npt.NDArray[np.bool_]
        True where band == "pta", shape (N,).
    in_lisa : npt.NDArray[np.bool_]
        True where band == "lisa", shape (N,).
    in_gap : npt.NDArray[np.bool_]
        True where band == "gap", shape (N,).
    """

    population: TNGPopulation
    f_isco_source_hz: npt.NDArray[np.float64]    # (N,) source frame
    f_isco_observer_hz: npt.NDArray[np.float64]  # (N,) observer frame = source/(1+z)
    band: npt.NDArray[np.str_]                   # (N,) "pta"/"lisa"/"gap"/"neither"
    in_pta: npt.NDArray[np.bool_]                # (N,)
    in_lisa: npt.NDArray[np.bool_]               # (N,)
    in_gap: npt.NDArray[np.bool_]                # (N,)

    @property
    def f_isco_hz(self) -> npt.NDArray[np.float64]:
        """Backward-compat alias for f_isco_source_hz (used by em_detectability)."""
        return self.f_isco_source_hz

    @property
    def n_pta(self) -> int:
        """Number of systems in the PTA band."""
        return int(np.sum(self.in_pta))

    @property
    def n_lisa(self) -> int:
        """Number of systems in the LISA band."""
        return int(np.sum(self.in_lisa))

    @property
    def n_gap(self) -> int:
        """Number of systems in the gap band."""
        return int(np.sum(self.in_gap))

    @property
    def n_neither(self) -> int:
        """Number of systems in neither band."""
        return int(np.sum(~self.in_pta & ~self.in_lisa & ~self.in_gap))


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------


def classify_bands(pop: TNGPopulation) -> GWClassification:
    """Classify every system in the population by GW band.

    Classification is applied to all systems including those that do not pass
    the quality cut; the quality mask (pop.passes_quality_cut) is left
    unmodified so that downstream code can combine the two masks freely.

    Parameters
    ----------
    pop : TNGPopulation
        Population derived from a TNGMergerCatalog.

    Returns
    -------
    GWClassification
        Arrays aligned 1-to-1 with pop.catalog arrays (length N = n_total).
    """
    # Source-frame ISCO frequency (diagnostic; also used by em_detectability
    # which applies its own (1+z) for the period conversion).
    f_source: npt.NDArray[np.float64] = compute_f_isco(pop.total_mass_msun)

    # Observer-frame ISCO frequency — used for band classification.
    # Detectors measure redshifted signals: f_obs = f_source / (1 + z).
    z: npt.NDArray[np.float64] = np.asarray(
        pop.catalog.redshift, dtype=np.float64
    )
    f_obs: npt.NDArray[np.float64] = f_source / (1.0 + z)

    in_pta: npt.NDArray[np.bool_] = (f_obs >= PTA_F_MIN_HZ) & (f_obs <= PTA_F_MAX_HZ)
    in_lisa: npt.NDArray[np.bool_] = (f_obs >= LISA_F_MIN_HZ) & (f_obs <= LISA_F_MAX_HZ)
    in_gap: npt.NDArray[np.bool_] = (f_obs > PTA_F_MAX_HZ) & (f_obs < LISA_F_MIN_HZ)

    # Build string band array using np.select (no Python loops)
    band: npt.NDArray[np.str_] = np.select(
        condlist=[in_pta, in_lisa, in_gap],
        choicelist=[
            np.full(f_obs.shape, GWBand.PTA.value),
            np.full(f_obs.shape, GWBand.LISA.value),
            np.full(f_obs.shape, GWBand.GAP.value),
        ],
        default=GWBand.NEITHER.value,
    )

    return GWClassification(
        population=pop,
        f_isco_source_hz=f_source,
        f_isco_observer_hz=f_obs,
        band=band,
        in_pta=in_pta,
        in_lisa=in_lisa,
        in_gap=in_gap,
    )
