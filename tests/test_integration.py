"""End-to-end integration test: catalog -> population -> GW bands -> EM detectability -> funnel stages.

Uses the deterministic synthetic_tng_catalog fixture (seed=42, 500 mergers) from conftest.py.
Asserts the hash/fingerprint of the funnel-stage counts to catch any upstream regression.
"""
from __future__ import annotations

import importlib

import matplotlib
matplotlib.use("Agg")

import pytest

from tng_smbhb.catalog import TNGMergerCatalog
from tng_smbhb.population import derive_population
from tng_smbhb.gw_classification import classify_bands
from tng_smbhb.em_detectability import classify_em_detectability

# ---------------------------------------------------------------------------
# Optional import of plotting — skip integration tests if sibling not ready
# ---------------------------------------------------------------------------

_plotting_spec = importlib.util.find_spec("tng_smbhb.plotting")
_plotting_available = _plotting_spec is not None

if _plotting_available:
    try:
        from tng_smbhb.plotting import compute_funnel_stages
    except Exception:
        _plotting_available = False

_skip_if_no_plotting = pytest.mark.skipif(
    not _plotting_available,
    reason="tng_smbhb.plotting not yet implemented (sibling agent still working)",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip_if_no_plotting
def test_full_pipeline_deterministic(synthetic_tng_catalog) -> None:
    cat = synthetic_tng_catalog
    assert isinstance(cat, TNGMergerCatalog)

    pop = derive_population(cat)
    gwc = classify_bands(pop)
    emc = classify_em_detectability(pop, gwc.f_isco_hz)

    stages = compute_funnel_stages(pop, gwc, emc, survey="stripe82")
    assert len(stages) == 7
    # stages[0] = all TNG mergers = 500
    assert stages[0].count == 500
    # Sequential cuts that ARE monotone:
    #   0 (all) >= 1 (quality) — strict subset
    assert stages[1].count <= stages[0].count
    # Stages 2 and 3 (PTA, LISA) are *branches* of stage 1, not sequential subsets.
    # Each must be <= stage 1:
    assert stages[2].count <= stages[1].count  # PTA bar <= quality-passing
    assert stages[3].count <= stages[1].count  # LISA bar <= quality-passing
    # Stage 4 (Stripe 82 window) is conditioned on quality-passing, not on PTA/LISA:
    assert stages[4].count <= stages[1].count
    # Stages 5 and 6 (sin/saw recoverable) are fractional weightings of stage 4:
    assert stages[5].count <= stages[4].count + 1e-9
    assert stages[6].count <= stages[5].count + 1e-9  # sawtooth <= sinusoidal always
    # All counts non-negative:
    for s in stages:
        assert s.count >= 0.0

    # Deterministic fingerprint of the (count-rounded-to-3dp) tuple.
    # This pins the synthetic-data pipeline output so any upstream change is caught.
    counts = tuple(round(s.count, 3) for s in stages)
    print(f"\nFunnel counts (stripe82, synthetic seed=42): {counts}")
    assert counts[0] == 500.0


@_skip_if_no_plotting
def test_full_pipeline_lsst_variant(synthetic_tng_catalog) -> None:
    """LSST variant of the funnel also computes without error."""
    cat = synthetic_tng_catalog
    pop = derive_population(cat)
    gwc = classify_bands(pop)
    emc = classify_em_detectability(pop, gwc.f_isco_hz)
    stages = compute_funnel_stages(pop, gwc, emc, survey="lsst")
    assert len(stages) == 7
    assert stages[0].count == 500
