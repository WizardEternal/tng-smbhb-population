"""IllustrisTNG SMBHB merger catalog → multi-messenger gap plot pipeline.

This package loads the IllustrisTNG ``blackhole_mergers.hdf5`` catalog,
derives population-level quantities (chirp mass, redshift, orbital period),
classifies each merger by its gravitational-wave band (PTA / LISA / gap /
neither) and electromagnetic detectability (Lin, Charisi & Haiman 2026),
and renders the multi-messenger **gap plot** (figure #9 of the portfolio).

Modules
-------
catalog : Load TNG HDF5 into structured arrays (unit conversions included).
population : Derived quantities + quality cuts (seed-mass exclusion).
gw_classification : PTA / LISA / gap band classifier for each merger.
em_detectability : Lin+2026 Lomb-Scargle recovery fractions applied to catalog.
plotting : Distribution plots + the gap plot (the visual punchline).

See ``README.md`` for the softening-scale caveat: TNG "mergers" occur at
the gravitational softening length (~0.7 kpc), not at physical parsec-scale
separations.  All GW band classifications are conditional on the final
parsec problem being solved.
"""

from __future__ import annotations

__version__ = "0.1.0"
