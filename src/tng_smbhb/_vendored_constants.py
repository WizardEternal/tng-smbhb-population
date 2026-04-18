# =====================================================================
# VENDORED from smbhb-inspiral v0.1.0 (src/smbhb_inspiral/constants.py).
# Do not edit here. To update, copy the source file verbatim and update
# the version number above. See tng-smbhb-population/EXECUTION notes
# for vendoring rationale (no pip cross-import between sibling repos).
# =====================================================================
"""Physical constants for the SMBHB inspiral simulator.

All constants are extracted from :mod:`astropy.constants` and stored as plain
Python :class:`float` values in SI units (or CGS where noted).

**Why plain floats instead of** ``astropy.Quantity`` **objects?**

``astropy.Quantity`` carries unit metadata and performs unit-checking
arithmetic, which is invaluable for dimensional analysis during development.
However, inside tight ODE integrators (e.g. ``scipy.integrate.solve_ivp``) the
overhead of repeated Quantity arithmetic — attribute lookups, unit propagation,
``__array_ufunc__`` dispatch — adds up to a measurable slowdown per right-hand-
side evaluation.  Extracting bare floats once at import time keeps the inner
loop as fast as raw NumPy while still documenting units through inline comments
and type annotations.

Values are taken from ``astropy`` 7.2.0 (CODATA 2018 / IAU 2015 where
applicable).  Re-run the snippet below to regenerate if the astropy version
changes::

    import astropy.constants as const
    import astropy.units as u
    print(const.G.si.value)          # G
    print(const.c.si.value)          # c
    print(const.M_sun.si.value)      # M_SUN
    print((1*u.pc).to(u.m).value)    # PC
    print((1*u.Mpc).to(u.m).value)   # MPC
    print((1*u.yr).to(u.s).value)    # YR
    print(const.G.cgs.value)         # G_CGS
    print(const.c.cgs.value)         # c_CGS
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# SI constants
# ---------------------------------------------------------------------------

G: float = 6.6743e-11
"""Newtonian gravitational constant  [m^3 kg^-1 s^-2]  (CODATA 2018)."""

c: float = 299_792_458.0
"""Speed of light in vacuum  [m s^-1]  (exact, SI definition)."""

M_SUN: float = 1.988409870698051e30
"""Solar mass  [kg]  (IAU 2015 nominal solar mass parameter / G)."""

PC: float = 3.085677581491367e16
"""Parsec  [m]  (IAU 2015: 1 pc = 648000/pi AU)."""

MPC: float = 3.085677581491367e22
"""Megaparsec  [m]  (= 1e6 * PC)."""

YR: float = 31_557_600.0
"""Julian year  [s]  (= 365.25 × 86400 s, exact by IAU convention)."""

# ---------------------------------------------------------------------------
# CGS constants  (for formulae inherited from the pulsar-timing / GW literature
# that work in Gaussian CGS units)
# ---------------------------------------------------------------------------

G_CGS: float = 6.674299999999999e-08
"""Newtonian gravitational constant  [cm^3 g^-1 s^-2]  (CODATA 2018)."""

c_CGS: float = 29_979_245_800.0
"""Speed of light in vacuum  [cm s^-1]  (exact)."""
