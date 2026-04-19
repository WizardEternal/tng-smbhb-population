# =====================================================================
# VENDORED from smbhb-inspiral v0.1.0 (src/smbhb_inspiral/physics.py).
# Do not edit here. To update, copy the source file verbatim and update
# the version number above. See tng-smbhb-population/EXECUTION notes
# for vendoring rationale (no pip cross-import between sibling repos).
# =====================================================================
"""Core post-Newtonian inspiral physics for the SMBHB gravitational wave simulator.

This module implements the frequency-domain post-Newtonian (PN) ordinary
differential equation (ODE) for the gravitational-wave frequency evolution of
a quasi-circular supermassive black-hole binary (SMBHB), together with the
orbital phase.  The integrator is :func:`scipy.integrate.solve_ivp` using the
explicit Runge-Kutta DOP853 scheme, which provides 8th-order accuracy with 5th-
and 3rd-order error estimates.

Physical foundations
--------------------
The leading-order (0PN) Newtonian quadrupole formula is due to Peters (1964).
The 1PN correction to the frequency derivative follows Blanchet (2014), Living
Reviews in Relativity **17**, 2 (hereafter B14), Section 7.1.

All quantities inside the ODE right-hand side are in SI units (kg, m, s, Hz).
Public API functions accept masses in solar masses and frequencies in Hz.

References
----------
Peters, P. C. 1964, Phys. Rev., 136, B1224.
    https://doi.org/10.1103/PhysRev.136.B1224

Blanchet, L. 2014, Living Rev. Relativity, 17, 2 (B14).
    https://doi.org/10.12942/lrr-2014-2
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from ._vendored_constants import G, c, M_SUN


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InspiralTrajectory:
    """Complete inspiral trajectory from ODE integration.

    Attributes
    ----------
    t : npt.NDArray[np.float64]
        Coordinate time array  [s].
    f_gw : npt.NDArray[np.float64]
        Gravitational-wave frequency  [Hz].
    f_orb : npt.NDArray[np.float64]
        Orbital frequency  [Hz].  Equal to ``f_gw / 2`` because the
        dominant GW harmonic is at twice the orbital frequency.
    a : npt.NDArray[np.float64]
        Orbital semi-major axis (separation)  [m], derived via Kepler's
        third law at each time step.
    v_over_c : npt.NDArray[np.float64]
        Dimensionless orbital velocity ``v / c``.
    phi : npt.NDArray[np.float64]
        Orbital phase  [rad], integrated from the ODE.
    chirp_mass_msun : float
        Chirp mass  [M_sun].
    total_mass_msun : float
        Total mass  ``m1 + m2``  [M_sun].
    eta : float
        Symmetric mass ratio  ``eta = m1*m2 / (m1+m2)^2`` (dimensionless,
        range ``(0, 1/4]``).
    pn_order : int
        Post-Newtonian order used in the integration (0 or 1).
    """

    t: npt.NDArray[np.float64]
    f_gw: npt.NDArray[np.float64]
    f_orb: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]
    v_over_c: npt.NDArray[np.float64]
    phi: npt.NDArray[np.float64]
    chirp_mass_msun: float
    total_mass_msun: float
    eta: float
    pn_order: int


# ---------------------------------------------------------------------------
# Public utility functions
# ---------------------------------------------------------------------------


def chirp_mass(m1: float, m2: float) -> float:
    """Compute the chirp mass of a binary system.

    The chirp mass is the combination of component masses that governs the
    leading-order (0PN) gravitational-wave phase evolution.

    Parameters
    ----------
    m1 : float
        Primary mass  [M_sun].  Must be positive.
    m2 : float
        Secondary mass  [M_sun].  Must be positive.

    Returns
    -------
    float
        Chirp mass  [M_sun].

    Notes
    -----
    The chirp mass is defined as

    .. math::

        \\mathcal{M}_c = \\frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}}

    which is manifestly symmetric under the exchange ``m1 <-> m2``.

    References
    ----------
    Peters & Mathews 1963, Phys. Rev. 131, 435, Eq. (3.16).
    """
    if m1 <= 0.0 or m2 <= 0.0:
        raise ValueError(
            f"Component masses must be positive; got m1={m1}, m2={m2}."
        )
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def symmetric_mass_ratio(m1: float, m2: float) -> float:
    """Compute the symmetric mass ratio (reduced mass ratio) of a binary.

    Parameters
    ----------
    m1 : float
        Primary mass  [M_sun].  Must be positive.
    m2 : float
        Secondary mass  [M_sun].  Must be positive.

    Returns
    -------
    float
        Symmetric mass ratio  ``eta = m1*m2 / (m1+m2)^2`` (dimensionless).
        Equals ``1/4`` for an equal-mass binary; approaches ``0`` in the
        test-particle (extreme mass-ratio) limit.

    Notes
    -----
    .. math::

        \\eta = \\frac{m_1 m_2}{(m_1 + m_2)^2}
    """
    if m1 <= 0.0 or m2 <= 0.0:
        raise ValueError(
            f"Component masses must be positive; got m1={m1}, m2={m2}."
        )
    m_tot = m1 + m2
    return (m1 * m2) / (m_tot * m_tot)


def f_isco(m_total_msun: float) -> float:
    """Gravitational-wave frequency at the innermost stable circular orbit (ISCO).

    For a Schwarzschild (non-spinning) black hole of total mass ``M``, the ISCO
    sits at ``r_ISCO = 6 G M / c^2``.  Kepler's third law then gives the
    orbital frequency, and the GW frequency is twice that.

    Parameters
    ----------
    m_total_msun : float
        Total binary mass  [M_sun].  Must be positive.

    Returns
    -------
    float
        GW frequency at ISCO  [Hz].

    Notes
    -----
    The ISCO GW frequency is

    .. math::

        f_{\\rm ISCO} = \\frac{c^3}{6^{3/2}\\,\\pi\\, G\\, M_{\\rm tot}}

    This is used as the default termination frequency for
    :func:`integrate_inspiral`.

    References
    ----------
    Peters, P. C. 1964, Phys. Rev. 136, B1224.
    Misner, Thorne & Wheeler 1973, *Gravitation*, §33.5.
    """
    if m_total_msun <= 0.0:
        raise ValueError(
            f"Total mass must be positive; got m_total_msun={m_total_msun}."
        )
    m_tot_kg = m_total_msun * M_SUN
    # 6^(3/2) = 6 * sqrt(6)
    six_to_three_halves = 6.0 ** 1.5
    return c ** 3 / (six_to_three_halves * math.pi * G * m_tot_kg)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------


def peters_rhs(
    t: float,
    y: npt.NDArray[np.float64],
    chirp_mass_kg: float,
    eta: float,
    total_mass_kg: float,
    pn_order: int = 1,
) -> npt.NDArray[np.float64]:
    """Right-hand side of the post-Newtonian inspiral ODE.

    The state vector is ``y = [f_GW (Hz), phi (rad)]`` where ``f_GW`` is
    the gravitational-wave frequency and ``phi`` is the *orbital* phase.

    Parameters
    ----------
    t : float
        Current time  [s].  Not used explicitly (autonomous ODE) but
        required by the :func:`scipy.integrate.solve_ivp` interface.
    y : npt.NDArray[np.float64], shape (2,)
        State vector ``[f_GW, phi]`` at time ``t``.
    chirp_mass_kg : float
        Chirp mass  [kg].
    eta : float
        Symmetric mass ratio (dimensionless).
    total_mass_kg : float
        Total binary mass  [kg].  Used only for the 1PN correction.
    pn_order : int, optional
        Post-Newtonian order.  ``0`` keeps only the leading quadrupole term;
        ``1`` adds the first post-Newtonian correction.  Default is ``1``.

    Returns
    -------
    npt.NDArray[np.float64], shape (2,)
        Array ``[df_GW/dt, dphi/dt]`` evaluated at ``(t, y)``.

    Notes
    -----
    **0PN (Newtonian quadrupole, Peters 1964):**

    .. math::

        \\frac{df}{dt}\\bigg|_{\\rm 0PN}
          = \\frac{96}{5}\\,\\pi^{8/3}
            \\left(\\frac{G \\mathcal{M}_c}{c^3}\\right)^{5/3}
            f^{11/3}

    **1PN correction factor** (Blanchet 2014, B14 Eq. 234 / Sec. 7.1):

    .. math::

        F_{\\rm 1PN} = 1 + \\left[
          -\\left(\\frac{743}{336} + \\frac{11}{4}\\,\\eta\\right)
        \\right] x

    where the dimensionless PN parameter is

    .. math::

        x = \\left(\\frac{\\pi G M_{\\rm tot} f_{\\rm GW}}{c^3}\\right)^{2/3}

    **Orbital phase:**

    Because the dominant GW harmonic is at twice the orbital frequency
    (``f_GW = 2 f_orb``), the orbital angular velocity is
    ``Omega_orb = pi f_GW``, and

    .. math::

        \\frac{d\\phi}{dt} = \\pi\\, f_{\\rm GW}

    References
    ----------
    Peters, P. C. 1964, Phys. Rev. 136, B1224, Eq. (5.14).
    Blanchet, L. 2014, Living Rev. Relativity 17, 2 (B14), Sec. 7.1,
        Eq. (234).
    """
    f_gw: float = y[0]
    # phi  = y[1]  # not needed for RHS computation

    # ------------------------------------------------------------------
    # Pre-compute recurring combinations
    # ------------------------------------------------------------------

    # Dimensionless chirp mass in geometric units: G * M_c / c^3  [s]
    gm_chirp_over_c3: float = G * chirp_mass_kg / c ** 3

    # 0PN frequency derivative (Peters 1964, Eq. 5.14)
    # df/dt = (96/5) * pi^(8/3) * (G*Mc/c^3)^(5/3) * f^(11/3)
    coeff_0pn: float = (96.0 / 5.0) * math.pi ** (8.0 / 3.0) * gm_chirp_over_c3 ** (5.0 / 3.0)
    df_dt: float = coeff_0pn * f_gw ** (11.0 / 3.0)

    # ------------------------------------------------------------------
    # 1PN correction
    # ------------------------------------------------------------------
    if pn_order >= 1:
        # PN parameter x = (pi * G * M_tot * f_GW / c^3)^(2/3)
        # Uses TOTAL mass, not chirp mass (B14, Sec. 3.1)
        gm_tot_over_c3: float = G * total_mass_kg / c ** 3
        x: float = (math.pi * gm_tot_over_c3 * f_gw) ** (2.0 / 3.0)

        # 1PN amplitude correction (B14 Eq. 234, spin-zero limit)
        # Coefficient: -(743/336 + 11/4 * eta)
        pn1_coeff: float = -(743.0 / 336.0 + (11.0 / 4.0) * eta)
        f_1pn: float = 1.0 + pn1_coeff * x

        df_dt *= f_1pn

    # ------------------------------------------------------------------
    # Orbital phase rate:  dphi/dt = pi * f_GW
    # (since f_orb = f_GW/2,  Omega_orb = 2*pi*f_orb = pi*f_GW)
    # ------------------------------------------------------------------
    dphi_dt: float = math.pi * f_gw

    return np.array([df_dt, dphi_dt], dtype=np.float64)


# ---------------------------------------------------------------------------
# Main integrator
# ---------------------------------------------------------------------------


def integrate_inspiral(
    m1: float,
    m2: float,
    f0: float,
    f_stop: float | None = None,
    t_max: float | None = None,
    pn_order: int = 1,
    rtol: float = 1e-10,
    atol: float = 1e-14,
) -> InspiralTrajectory:
    """Integrate the post-Newtonian SMBHB inspiral from an initial GW frequency.

    The ODE is integrated forward in time from ``f0`` until the GW frequency
    reaches ``f_stop`` (default: ISCO frequency) or until ``t_max`` is
    exceeded, whichever comes first.

    Parameters
    ----------
    m1 : float
        Primary mass  [M_sun].  Must be positive.
    m2 : float
        Secondary mass  [M_sun].  Must be positive.
    f0 : float
        Initial gravitational-wave frequency  [Hz].  Must be positive and
        less than ``f_stop``.
    f_stop : float or None, optional
        Terminal GW frequency  [Hz].  The integrator stops when
        ``f_GW >= f_stop``.  Defaults to the Schwarzschild ISCO frequency
        :func:`f_isco`.
    t_max : float or None, optional
        Maximum integration time  [s].  If ``None`` (default), an upper
        bound of ``1.05 * analytic_t_merge_circular(m1, m2, f0)`` is used.
        The terminal event (``f_GW >= f_stop``) will fire before this limit
        for any physically reasonable configuration.
    pn_order : int, optional
        Post-Newtonian order (0 or 1).  Default is ``1``.
    rtol : float, optional
        Relative tolerance for the ODE solver.  Default ``1e-10``.
    atol : float, optional
        Absolute tolerance for the ODE solver.  Default ``1e-14``.

    Returns
    -------
    InspiralTrajectory
        Dataclass containing the full trajectory and derived orbital
        quantities.  All arrays are aligned in time.

    Raises
    ------
    ValueError
        If masses or frequencies are non-positive, or if ``f0 >= f_stop``.
    RuntimeError
        If the ODE integrator fails to converge.

    Notes
    -----
    The integrator uses the DOP853 scheme (Dormand & Prince 1980), an
    explicit 8th-order Runge-Kutta method with adaptive step-size control.
    The dense output is evaluated on a uniformly log-spaced grid of 10 000
    points to provide smooth trajectories for downstream analysis.

    The post-integration derived quantities are:

    * Orbital frequency:  ``f_orb = f_gw / 2``
    * Orbital separation via Kepler III:

      .. math::

          a = \\left(\\frac{G M_{\\rm tot}}{4 \\pi^2 f_{\\rm orb}^2}\\right)^{1/3}

    * Orbital velocity:

      .. math::

          v/c = \\left(\\frac{\\pi G M_{\\rm tot} f_{\\rm GW}}{c^3}\\right)^{1/3}

    References
    ----------
    Peters, P. C. 1964, Phys. Rev. 136, B1224.
    Blanchet, L. 2014, Living Rev. Relativity 17, 2, Sec. 7.1.
    Dormand, J. R. & Prince, P. J. 1980, J. Comput. Appl. Math. 6, 19.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if m1 <= 0.0 or m2 <= 0.0:
        raise ValueError(
            f"Component masses must be positive; got m1={m1}, m2={m2}."
        )
    if f0 <= 0.0:
        raise ValueError(f"Initial GW frequency must be positive; got f0={f0}.")

    # ------------------------------------------------------------------
    # Derived masses in SI
    # ------------------------------------------------------------------
    m1_kg: float = m1 * M_SUN
    m2_kg: float = m2 * M_SUN
    m_tot_msun: float = m1 + m2
    m_tot_kg: float = m_tot_msun * M_SUN

    mc_msun: float = chirp_mass(m1, m2)
    mc_kg: float = mc_msun * M_SUN
    eta: float = symmetric_mass_ratio(m1, m2)

    # ------------------------------------------------------------------
    # Termination frequency
    # ------------------------------------------------------------------
    f_isco_val: float = f_isco(m_tot_msun)
    if f_stop is None:
        f_stop = f_isco_val

    if f0 >= f_stop:
        raise ValueError(
            f"Initial frequency f0={f0:.6g} Hz must be less than f_stop={f_stop:.6g} Hz."
        )

    # ------------------------------------------------------------------
    # Integration time span
    # ------------------------------------------------------------------
    if t_max is None:
        # Use 1.05x the analytic 0PN merger time as the upper bound.
        # The terminal event (f_GW >= f_stop) fires well before this limit.
        # Using a value much larger than t_analytic causes the integrator's
        # adaptive step control to degrade because t_span becomes so large
        # that consecutive floating-point values are indistinguishable.
        t_analytic: float = analytic_t_merge_circular(m1, m2, f0)
        t_max = 1.05 * t_analytic

    t_span: tuple[float, float] = (0.0, t_max)
    y0: npt.NDArray[np.float64] = np.array([f0, 0.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Terminal event: stop when f_GW reaches f_stop
    # ------------------------------------------------------------------
    def _event_f_stop(
        t: float,
        y: npt.NDArray[np.float64],
        chirp_mass_kg: float,
        eta: float,
        total_mass_kg: float,
        pn_order: int,
    ) -> float:
        return y[0] - f_stop  # crosses zero from below

    _event_f_stop.terminal = True   # type: ignore[attr-defined]
    _event_f_stop.direction = +1    # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Integrate
    # ------------------------------------------------------------------
    sol = solve_ivp(
        fun=peters_rhs,
        t_span=t_span,
        y0=y0,
        method="DOP853",
        args=(mc_kg, eta, m_tot_kg, pn_order),
        events=_event_f_stop,
        dense_output=True,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(
            f"ODE integration failed: {sol.message}"
        )

    # ------------------------------------------------------------------
    # Evaluate on a smooth log-spaced output grid (10 000 points)
    # ------------------------------------------------------------------
    t_end: float = sol.t[-1]
    t_start: float = sol.t[0]

    # Use log-spacing so that early (slow) and late (fast) evolution are
    # both well-sampled.  Add a small offset to avoid log(0).
    if t_end > t_start:
        t_eval: npt.NDArray[np.float64] = np.geomspace(
            max(t_start, 1.0e-6 * t_end),
            t_end,
            num=10_000,
        )
        # Prepend t=0 (initial condition) if it was clipped above
        if t_eval[0] > t_start:
            t_eval = np.concatenate([[t_start], t_eval])
    else:
        # Edge case: integrator stopped immediately
        t_eval = np.array([t_start, t_end], dtype=np.float64)

    y_eval: npt.NDArray[np.float64] = sol.sol(t_eval)  # shape (2, N)
    f_gw_arr: npt.NDArray[np.float64] = y_eval[0]
    phi_arr: npt.NDArray[np.float64] = y_eval[1]

    # Clip any tiny numerical overshoots beyond f_stop
    f_gw_arr = np.minimum(f_gw_arr, f_stop)
    # Ensure frequencies remain physical (positive)
    f_gw_arr = np.maximum(f_gw_arr, 0.0)

    # ------------------------------------------------------------------
    # Derived orbital quantities
    # ------------------------------------------------------------------
    f_orb_arr: npt.NDArray[np.float64] = f_gw_arr / 2.0

    # Orbital separation via Kepler's third law:
    #   a^3 = G M_tot / (4 pi^2 f_orb^2)
    # => a = (G M_tot / (4 pi^2 f_orb^2))^(1/3)
    four_pi2: float = 4.0 * math.pi ** 2
    a_arr: npt.NDArray[np.float64] = (
        G * m_tot_kg / (four_pi2 * f_orb_arr ** 2)
    ) ** (1.0 / 3.0)

    # Dimensionless orbital velocity:
    #   v_orb = 2*pi*f_orb*a, with Kepler a^3 = G M_tot / (4 pi^2 f_orb^2)
    #   => v_orb^3 = 2 pi G M_tot f_orb = pi G M_tot f_GW   (f_orb = f_GW/2)
    #   => v/c = (pi G M_tot f_GW / c^3)^(1/3)
    # At f_ISCO this correctly gives v/c = 1/sqrt(6) ~ 0.408.
    v_over_c_arr: npt.NDArray[np.float64] = (
        math.pi * G * m_tot_kg * f_gw_arr / (c ** 3)
    ) ** (1.0 / 3.0)

    # ------------------------------------------------------------------
    # Assemble and return trajectory
    # ------------------------------------------------------------------
    return InspiralTrajectory(
        t=t_eval,
        f_gw=f_gw_arr,
        f_orb=f_orb_arr,
        a=a_arr,
        v_over_c=v_over_c_arr,
        phi=phi_arr,
        chirp_mass_msun=mc_msun,
        total_mass_msun=m_tot_msun,
        eta=eta,
        pn_order=pn_order,
    )


# ---------------------------------------------------------------------------
# Analytic time-to-merger (0PN, Peters 1964)
# ---------------------------------------------------------------------------


def analytic_t_merge_circular(m1: float, m2: float, f0: float) -> float:
    """Analytic 0PN time to merger for a circular orbit.

    Uses the Peters (1964) formula expressed in terms of the initial orbital
    separation, which is obtained from the initial GW frequency via Kepler's
    third law.

    Parameters
    ----------
    m1 : float
        Primary mass  [M_sun].  Must be positive.
    m2 : float
        Secondary mass  [M_sun].  Must be positive.
    f0 : float
        Initial gravitational-wave frequency  [Hz].  Must be positive.

    Returns
    -------
    float
        Approximate time to merger  [s]  (0PN, circular orbit).

    Notes
    -----
    From Kepler's third law the initial orbital separation is

    .. math::

        a_0 = \\left(
            \\frac{G M_{\\rm tot}}{4 \\pi^2 f_{\\rm orb,0}^2}
        \\right)^{1/3}

    where ``f_orb,0 = f0 / 2``.  Peters (1964), Eq. (5.14), then gives

    .. math::

        t_{\\rm merge} = \\frac{5}{256}
            \\frac{c^5\\, a_0^4}{G^3\\, m_1\\, m_2\\, (m_1 + m_2)}

    This is the exact 0PN result for a circular orbit starting at ``a_0``
    with zero eccentricity.

    References
    ----------
    Peters, P. C. 1964, Phys. Rev. 136, B1224, Eq. (5.14).
    """
    if m1 <= 0.0 or m2 <= 0.0:
        raise ValueError(
            f"Component masses must be positive; got m1={m1}, m2={m2}."
        )
    if f0 <= 0.0:
        raise ValueError(f"Initial GW frequency must be positive; got f0={f0}.")

    m1_kg: float = m1 * M_SUN
    m2_kg: float = m2 * M_SUN
    m_tot_kg: float = m1_kg + m2_kg

    # Initial orbital frequency
    f_orb_0: float = f0 / 2.0

    # Initial orbital separation via Kepler III
    a0: float = (G * m_tot_kg / (4.0 * math.pi ** 2 * f_orb_0 ** 2)) ** (1.0 / 3.0)

    # Peters (1964) Eq. (5.14):  t = (5/256) * c^5 * a0^4 / (G^3 * m1 * m2 * M_tot)
    t_merge: float = (
        (5.0 / 256.0)
        * c ** 5
        * a0 ** 4
        / (G ** 3 * m1_kg * m2_kg * m_tot_kg)
    )
    return t_merge
