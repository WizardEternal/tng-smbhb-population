# =====================================================================
# VENDORED from smbhb-inspiral v0.1.0 (src/smbhb_inspiral/em_detectability.py).
# Do not edit here. To update, copy the source file verbatim and update
# the version number above. See tng-smbhb-population/EXECUTION notes
# for vendoring rationale (no pip cross-import between sibling repos).
# =====================================================================
"""Electromagnetic survey recoverability of SMBHB systems.

This module predicts whether a given supermassive black-hole binary (SMBHB)
would be recovered by optical variability surveys using the Lomb-Scargle (LS)
periodogram, based on the published aggregate recovery fractions of

    Lin, Charisi & Haiman 2026, ApJ 997, 316
    DOI 10.3847/1538-4357/ae29a7

In particular, Table 1 and Section 3.1 of that paper supply the following
LS recovery fractions (approximated to the nearest percent as quoted):

+------------+------------+------------+------------+
| Signal     | PTF-like   | Idealized  | LSST-like  |
+============+============+============+============+
| Sinusoidal | ~45 %      | ~24 %      | ~23 %      |
| Sawtooth   |  ~9 %      |  ~1 %      |  ~1 %      |
+------------+------------+------------+------------+

The paper contains the following notable observation about the significance of
missed detections:

    "Previous searches, including the one in M. Charisi et al. (2016), must
    have missed a significant fraction of periodic signals."
    — Lin, Charisi & Haiman 2026, ApJ 997, 316

Aggregate-fraction caveat
-------------------------
We apply aggregate recovery fractions, not individualized injection-recovery.
Per-system rates depend on DRW parameters, signal amplitude, and cadence,
which would require a separate simulation campaign.

Usage
-----
>>> from tng_smbhb._vendored_em_detectability import classify_system
>>> result = classify_system(m_total_msun=1e8, f_gw_hz=1e-8, z=0.5)
>>> result.in_lsst_window
True
>>> result.recovery_sinusoidal_lsst
0.23
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._vendored_constants import G, c, M_SUN  # noqa: F401 — c kept for potential future use

# ---------------------------------------------------------------------------
# Published recovery fractions — Lin, Charisi & Haiman 2026, ApJ 997, 316
# Table 1 / Section 3.1.  DO NOT MODIFY without updating the citation.
# ---------------------------------------------------------------------------

RECOVERY_FRACTIONS: dict[str, dict[str, float]] = {
    "sinusoidal": {
        "ptf_like":  0.45,
        "idealized": 0.24,
        "lsst_like": 0.23,
    },
    "sawtooth": {
        "ptf_like":  0.09,
        "idealized": 0.01,
        "lsst_like": 0.01,
    },
}
"""LS recovery fractions from Lin, Charisi & Haiman 2026, ApJ 997, 316.

Outer key: signal shape (``"sinusoidal"`` or ``"sawtooth"``).
Inner key: survey cadence type (``"ptf_like"``, ``"idealized"``,
``"lsst_like"``).
Values are dimensionless fractions in [0, 1].
"""

# ---------------------------------------------------------------------------
# Survey window definitions
#
# P_max is set by the ~3-cycle requirement: at least three full oscillations
# must fit within the survey baseline for the LS periodogram to reliably
# identify a period.  P_max ≈ baseline / 3.
#
# P_min is cadence-limited: the Nyquist-like lower bound requires several
# observations per cycle.  The values below follow the cadence assumptions of
# Lin, Charisi & Haiman 2026 and the original PTF/Stripe82 analyses.
# ---------------------------------------------------------------------------

SURVEY_WINDOWS: dict[str, dict[str, float]] = {
    "stripe82": {
        "P_min_days":    200.0,
        "P_max_days":   1100.0,
        "baseline_days": 3650.0,
        "cadence":       "ptf_like",
    },
    "ptf": {
        "P_min_days":    100.0,
        "P_max_days":    600.0,
        "baseline_days": 1825.0,
        "cadence":       "ptf_like",
    },
    "lsst": {
        "P_min_days":    100.0,
        "P_max_days":   1200.0,
        "baseline_days": 3650.0,
        "cadence":       "lsst_like",
    },
}
"""Sensitivity windows for each optical variability survey.

Keys
----
stripe82 : SDSS Stripe 82, ~10 yr baseline, PTF-like cadence.
ptf      : Palomar Transient Factory, ~5 yr baseline, PTF-like cadence.
lsst     : Vera C. Rubin Observatory / LSST, ~10 yr baseline, LSST cadence.

Per-survey sub-keys
-------------------
P_min_days : float
    Minimum detectable observer-frame period [days].  Set by cadence: the
    survey must sample at least several cycles per period.
P_max_days : float
    Maximum detectable observer-frame period [days].  Approximately
    ``baseline_days / 3`` to satisfy the 3-cycle requirement.
baseline_days : float
    Total photometric baseline of the survey [days].
cadence : str
    Key into :data:`RECOVERY_FRACTIONS` cadence tier
    (``"ptf_like"`` or ``"lsst_like"``).

Note: both Stripe 82 and PTF use the ``"ptf_like"`` recovery fractions
because their observing strategies are comparable (see Lin et al. 2026,
Section 2.2).
"""

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EMDetectabilityResult:
    """EM survey detectability classification for a single SMBHB system.

    All ``recovery_*`` fields are drawn directly from
    :data:`RECOVERY_FRACTIONS` (Lin, Charisi & Haiman 2026) when the system
    falls inside the corresponding survey window, and are 0.0 otherwise.

    Parameters
    ----------
    p_rest_days : float
        Rest-frame orbital period [days].
    p_obs_days : float
        Observer-frame orbital period [days], i.e. ``p_rest_days * (1 + z)``.
    in_stripe82_window : bool
        True if ``p_obs_days`` lies within the Stripe 82 sensitivity window.
    in_ptf_window : bool
        True if ``p_obs_days`` lies within the PTF sensitivity window.
    in_lsst_window : bool
        True if ``p_obs_days`` lies within the LSST sensitivity window.
    recovery_sinusoidal_stripe82 : float
        LS recovery fraction for a sinusoidal signal in Stripe 82 [0–1].
    recovery_sawtooth_stripe82 : float
        LS recovery fraction for a sawtooth signal in Stripe 82 [0–1].
    recovery_sinusoidal_ptf : float
        LS recovery fraction for a sinusoidal signal in PTF [0–1].
    recovery_sawtooth_ptf : float
        LS recovery fraction for a sawtooth signal in PTF [0–1].
    recovery_sinusoidal_lsst : float
        LS recovery fraction for a sinusoidal signal in LSST [0–1].
    recovery_sawtooth_lsst : float
        LS recovery fraction for a sawtooth signal in LSST [0–1].
    """

    p_rest_days: float
    p_obs_days: float
    in_stripe82_window: bool
    in_ptf_window: bool
    in_lsst_window: bool
    recovery_sinusoidal_stripe82: float   # 0–1
    recovery_sawtooth_stripe82: float     # 0–1
    recovery_sinusoidal_ptf: float        # 0–1
    recovery_sawtooth_ptf: float          # 0–1
    recovery_sinusoidal_lsst: float       # 0–1
    recovery_sawtooth_lsst: float         # 0–1


# ---------------------------------------------------------------------------
# Period / frequency utility functions
# ---------------------------------------------------------------------------

_SECONDS_PER_DAY: float = 86_400.0


def orbital_period_from_separation(
    m_total_msun: float,
    separation_m: float,
) -> float:
    """Compute the Keplerian orbital period from total mass and separation.

    Uses Kepler's third law:

    .. math::

        P = 2\\pi \\sqrt{\\frac{a^3}{G M_{\\rm tot}}}

    Parameters
    ----------
    m_total_msun : float
        Total binary mass [M_sun].
    separation_m : float
        Binary separation (semi-major axis) [m].

    Returns
    -------
    float
        Orbital period [s].

    Examples
    --------
    >>> orbital_period_from_separation(1e8, 1e13)  # doctest: +ELLIPSIS
    1...
    """
    m_total_kg: float = m_total_msun * M_SUN
    period_s: float = 2.0 * math.pi * math.sqrt(separation_m**3 / (G * m_total_kg))
    return period_s


def orbital_period_from_f_gw(f_gw_hz: float) -> float:
    """Compute the orbital period from the gravitational-wave frequency.

    The GW frequency is twice the orbital frequency for a quasi-circular
    binary:

    .. math::

        f_{\\rm GW} = 2 f_{\\rm orb}
        \\quad\\Rightarrow\\quad
        P_{\\rm orb} = \\frac{2}{f_{\\rm GW}}

    Parameters
    ----------
    f_gw_hz : float
        Gravitational-wave frequency [Hz].

    Returns
    -------
    float
        Orbital period [s].

    Raises
    ------
    ValueError
        If ``f_gw_hz`` is not positive.
    """
    if f_gw_hz <= 0.0:
        raise ValueError(f"f_gw_hz must be positive; got {f_gw_hz!r}")
    return 2.0 / f_gw_hz


def observer_frame_period(p_rest_s: float, z: float) -> float:
    """Convert a rest-frame orbital period to the observer frame.

    .. math::

        P_{\\rm obs} = P_{\\rm rest}\\,(1 + z)

    Parameters
    ----------
    p_rest_s : float
        Rest-frame orbital period [s].
    z : float
        Cosmological redshift (must be >= 0).

    Returns
    -------
    float
        Observer-frame orbital period [s].

    Raises
    ------
    ValueError
        If ``z`` is negative.
    """
    if z < 0.0:
        raise ValueError(f"Redshift z must be >= 0; got {z!r}")
    return p_rest_s * (1.0 + z)


# ---------------------------------------------------------------------------
# Survey-window and recovery-fraction helpers
# ---------------------------------------------------------------------------


def in_survey_window(p_obs_days: float, survey: str) -> bool:
    """Check whether an observer-frame period falls within a survey window.

    Parameters
    ----------
    p_obs_days : float
        Observer-frame orbital period [days].
    survey : str
        Survey name; must be a key of :data:`SURVEY_WINDOWS`
        (``"stripe82"``, ``"ptf"``, or ``"lsst"``).

    Returns
    -------
    bool
        True if ``P_min_days <= p_obs_days <= P_max_days`` for the survey.

    Raises
    ------
    ValueError
        If ``survey`` is not a recognised survey name.
    """
    if survey not in SURVEY_WINDOWS:
        valid = ", ".join(f'"{s}"' for s in SURVEY_WINDOWS)
        raise ValueError(
            f"Unknown survey {survey!r}. Valid choices are: {valid}."
        )
    window = SURVEY_WINDOWS[survey]
    return float(window["P_min_days"]) <= p_obs_days <= float(window["P_max_days"])


def recovery_fraction(
    signal_shape: Literal["sinusoidal", "sawtooth"],
    survey: str,
) -> float:
    """Look up the LS recovery fraction for a given signal shape and survey.

    The cadence tier is resolved from :data:`SURVEY_WINDOWS` and used to
    index into :data:`RECOVERY_FRACTIONS`.  This function does *not* check
    whether the period is inside the survey window; callers should use
    :func:`in_survey_window` for that guard, or use the high-level
    :func:`classify_system` which handles both.

    Parameters
    ----------
    signal_shape : {"sinusoidal", "sawtooth"}
        Shape of the assumed optical light-curve modulation.
    survey : str
        Survey name; must be a key of :data:`SURVEY_WINDOWS`.

    Returns
    -------
    float
        Recovery fraction in [0, 1] from Lin, Charisi & Haiman 2026,
        Table 1.

    Raises
    ------
    ValueError
        If ``survey`` is not recognised or ``signal_shape`` is invalid.
    """
    if survey not in SURVEY_WINDOWS:
        valid = ", ".join(f'"{s}"' for s in SURVEY_WINDOWS)
        raise ValueError(
            f"Unknown survey {survey!r}. Valid choices are: {valid}."
        )
    if signal_shape not in RECOVERY_FRACTIONS:
        valid_shapes = ", ".join(f'"{s}"' for s in RECOVERY_FRACTIONS)
        raise ValueError(
            f"Unknown signal_shape {signal_shape!r}. "
            f"Valid choices are: {valid_shapes}."
        )
    cadence: str = str(SURVEY_WINDOWS[survey]["cadence"])
    return RECOVERY_FRACTIONS[signal_shape][cadence]


# ---------------------------------------------------------------------------
# Main classification entry points
# ---------------------------------------------------------------------------

def _build_result(
    p_rest_s: float,
    p_obs_s: float,
) -> EMDetectabilityResult:
    """Internal helper: build an :class:`EMDetectabilityResult` from periods.

    Parameters
    ----------
    p_rest_s : float
        Rest-frame orbital period [s].
    p_obs_s : float
        Observer-frame orbital period [s].

    Returns
    -------
    EMDetectabilityResult
    """
    p_rest_days: float = p_rest_s / _SECONDS_PER_DAY
    p_obs_days: float = p_obs_s / _SECONDS_PER_DAY

    in_s82: bool = in_survey_window(p_obs_days, "stripe82")
    in_ptf: bool = in_survey_window(p_obs_days, "ptf")
    in_lsst: bool = in_survey_window(p_obs_days, "lsst")

    return EMDetectabilityResult(
        p_rest_days=p_rest_days,
        p_obs_days=p_obs_days,
        in_stripe82_window=in_s82,
        in_ptf_window=in_ptf,
        in_lsst_window=in_lsst,
        recovery_sinusoidal_stripe82=(
            recovery_fraction("sinusoidal", "stripe82") if in_s82 else 0.0
        ),
        recovery_sawtooth_stripe82=(
            recovery_fraction("sawtooth", "stripe82") if in_s82 else 0.0
        ),
        recovery_sinusoidal_ptf=(
            recovery_fraction("sinusoidal", "ptf") if in_ptf else 0.0
        ),
        recovery_sawtooth_ptf=(
            recovery_fraction("sawtooth", "ptf") if in_ptf else 0.0
        ),
        recovery_sinusoidal_lsst=(
            recovery_fraction("sinusoidal", "lsst") if in_lsst else 0.0
        ),
        recovery_sawtooth_lsst=(
            recovery_fraction("sawtooth", "lsst") if in_lsst else 0.0
        ),
    )


def classify_system(
    m_total_msun: float,
    f_gw_hz: float,
    z: float,
) -> EMDetectabilityResult:
    """Classify a SMBHB system's EM survey recoverability.

    This is the primary entry point for the module.  Given the total binary
    mass, current GW frequency, and source redshift, it:

    1. Computes the rest-frame orbital period from *f_gw* (assuming
       quasi-circular inspiral, so :math:`f_{\\rm GW} = 2 f_{\\rm orb}`).
    2. Converts to the observer-frame period via :math:`P_{\\rm obs} =
       P_{\\rm rest}(1+z)`.
    3. Determines whether the system falls within the sensitivity windows of
       Stripe 82, PTF, and LSST.
    4. Returns the corresponding LS recovery fractions (from Lin, Charisi &
       Haiman 2026) for both sinusoidal and sawtooth light-curve shapes, with
       0.0 for surveys where the period is outside the window.

    Parameters
    ----------
    m_total_msun : float
        Total binary mass [M_sun].  Used only indirectly (not needed when
        the period is derived from *f_gw*); retained for API consistency with
        :func:`classify_system_from_separation`.
    f_gw_hz : float
        Current gravitational-wave frequency [Hz].
    z : float
        Source redshift (>= 0).

    Returns
    -------
    EMDetectabilityResult
        Frozen dataclass with period information, window flags, and LS
        recovery fractions.

    Raises
    ------
    ValueError
        If ``f_gw_hz <= 0`` or ``z < 0``.

    Examples
    --------
    >>> res = classify_system(1e8, 1e-8, 0.3)
    >>> 0.0 <= res.recovery_sinusoidal_lsst <= 1.0
    True
    """
    p_rest_s: float = orbital_period_from_f_gw(f_gw_hz)
    p_obs_s: float = observer_frame_period(p_rest_s, z)
    return _build_result(p_rest_s, p_obs_s)


def classify_system_from_separation(
    m_total_msun: float,
    separation_m: float,
    z: float,
) -> EMDetectabilityResult:
    """Classify a SMBHB system's EM survey recoverability given its separation.

    Identical to :func:`classify_system` but derives the orbital period from
    the binary separation via Kepler's third law rather than from the GW
    frequency.

    Parameters
    ----------
    m_total_msun : float
        Total binary mass [M_sun].
    separation_m : float
        Binary separation (semi-major axis) [m].
    z : float
        Source redshift (>= 0).

    Returns
    -------
    EMDetectabilityResult
        Frozen dataclass with period information, window flags, and LS
        recovery fractions.

    Raises
    ------
    ValueError
        If ``z < 0``, ``m_total_msun <= 0``, or ``separation_m <= 0``.

    Examples
    --------
    >>> res = classify_system_from_separation(1e8, 1e13, 0.3)
    >>> isinstance(res.p_rest_days, float)
    True
    """
    if m_total_msun <= 0.0:
        raise ValueError(
            f"m_total_msun must be positive; got {m_total_msun!r}"
        )
    if separation_m <= 0.0:
        raise ValueError(
            f"separation_m must be positive; got {separation_m!r}"
        )
    p_rest_s: float = orbital_period_from_separation(m_total_msun, separation_m)
    p_obs_s: float = observer_frame_period(p_rest_s, z)
    return _build_result(p_rest_s, p_obs_s)
