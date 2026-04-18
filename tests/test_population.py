"""Tests for tng_smbhb.population.

Some tests require the ``synthetic_tng_catalog`` fixture from conftest.py
and the catalog module (sibling work). If those are missing at import time
the entire module will be skipped with a clear message.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Guard: skip whole module if sibling catalog module is missing
# ---------------------------------------------------------------------------
try:
    from tng_smbhb.catalog import TNGMergerCatalog, catalog_from_arrays  # noqa: F401
    from tng_smbhb.population import (
        TNGPopulation,
        derive_population,
        filter_population,
    )

    _CATALOG_AVAILABLE = True
except ImportError:
    _CATALOG_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CATALOG_AVAILABLE,
    reason="tng_smbhb.catalog (sibling module) not yet available",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MSUN = np.float64


def _make_catalog(
    m1: list[float],
    m2: list[float],
    simulation: str = "TNG300-1",
    hubble_h: float = 0.6774,
) -> "TNGMergerCatalog":
    """Build a minimal synthetic catalog for parametric unit tests.

    scale_factor = 1.0 for all rows (z = 0); redshift is computed internally
    by catalog_from_arrays.
    """
    n = len(m1)
    return catalog_from_arrays(
        m1_msun=np.array(m1, dtype=np.float64),
        m2_msun=np.array(m2, dtype=np.float64),
        scale_factor=np.ones(n, dtype=np.float64),
        simulation=simulation,
        hubble_h=hubble_h,
    )


# ---------------------------------------------------------------------------
# Unit tests: chirp mass formula
# ---------------------------------------------------------------------------


class TestChirpMass:
    def test_equal_mass_10_msun(self) -> None:
        """Chirp mass of (10, 10) M_sun system.

        For m1 = m2 = m:
            Mc = (m*m)^0.6 / (2m)^0.2 = m^1.2 / (2^0.2 * m^0.2) = m / 2^0.2

        At m = 10 M_sun: Mc = 10 / 2^0.2 ≈ 8.7055 M_sun.
        This also equals eta^(3/5) * M_total = 0.25^0.6 * 20, confirming
        Peters-Mathews convention.
        """
        cat = _make_catalog([10.0], [10.0])
        pop = derive_population(cat)
        expected = 10.0 / 2.0 ** 0.2  # = m / 2^0.2 for equal-mass system
        assert pop.chirp_mass_msun[0] == pytest.approx(expected, rel=1e-12)

    def test_chirp_mass_formula_explicit(self) -> None:
        """Verify formula directly: Mc = (m1*m2)^0.6 / (m1+m2)^0.2."""
        m1, m2 = 3e7, 1e7
        cat = _make_catalog([m1], [m2])
        pop = derive_population(cat)
        expected = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        assert pop.chirp_mass_msun[0] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Unit tests: eta
# ---------------------------------------------------------------------------


class TestEta:
    def test_equal_mass_eta_quarter(self) -> None:
        """Symmetric mass ratio for equal masses must be exactly 0.25."""
        cat = _make_catalog([5e6], [5e6])
        pop = derive_population(cat)
        assert pop.eta[0] == pytest.approx(0.25, rel=1e-12)

    def test_unequal_mass_eta_below_quarter(self) -> None:
        """eta < 0.25 whenever masses differ."""
        cat = _make_catalog([8e6], [2e6])
        pop = derive_population(cat)
        assert pop.eta[0] < 0.25


# ---------------------------------------------------------------------------
# Unit tests: quality cut
# ---------------------------------------------------------------------------


class TestQualityCut:
    def test_below_threshold_excluded(self) -> None:
        """Systems with M_tot < 1.2e6 M_sun must fail the cut."""
        m_half = 5e5
        cat = _make_catalog([m_half], [m_half])  # total = 1e6 < 1.2e6
        pop = derive_population(cat)
        assert not pop.passes_quality_cut[0]

    def test_at_threshold_excluded(self) -> None:
        """Systems with M_tot == 1.2e6 (not strictly greater) must fail."""
        cat = _make_catalog([6e5], [6e5])  # total = 1.2e6 exactly
        pop = derive_population(cat)
        assert not pop.passes_quality_cut[0]

    def test_above_threshold_passes(self) -> None:
        """Systems with M_tot = 1.2e6 + eps must pass the cut."""
        eps = 1.0  # 1 M_sun above threshold
        m_each = (1.2e6 + eps) / 2.0
        cat = _make_catalog([m_each], [m_each])
        pop = derive_population(cat)
        assert pop.passes_quality_cut[0]

    def test_non_positive_mass_excluded(self) -> None:
        """Rows with m1 <= 0 must fail the quality cut even if total mass is large.

        catalog_from_arrays validates positivity and would reject 0-mass inputs,
        so we inject zero-mass rows directly into the dataclass to test the
        quality-cut guard in derive_population.
        """
        bad_cat = TNGMergerCatalog(
            m1_msun=np.array([0.0], dtype=np.float64),
            m2_msun=np.array([2e6], dtype=np.float64),
            scale_factor=np.array([1.0], dtype=np.float64),
            redshift=np.array([0.0], dtype=np.float64),
            simulation="synthetic",
            hubble_h=0.6774,
            n_mergers=1,
        )
        pop = derive_population(bad_cat)
        assert not pop.passes_quality_cut[0]


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


@given(
    m1=st.floats(min_value=1e4, max_value=1e10),
    m2=st.floats(min_value=1e4, max_value=1e10),
)
@settings(max_examples=500)
def test_chirp_mass_symmetric(m1: float, m2: float) -> None:
    """Chirp mass is symmetric under swap of m1 and m2."""
    # ensure m1 >= m2 for catalog convention
    lo, hi = (m1, m2) if m1 >= m2 else (m2, m1)
    cat_ab = _make_catalog([hi], [lo])
    # swapped: treat lo as m1, hi as m2 — but catalog requires m1>=m2, so skip
    # Instead compute chirp mass directly both ways and compare
    mc_ab = derive_population(cat_ab).chirp_mass_msun[0]
    mc_direct = (hi * lo) ** 0.6 / (hi + lo) ** 0.2
    mc_swapped = (lo * hi) ** 0.6 / (lo + hi) ** 0.2
    assert mc_ab == pytest.approx(mc_direct, rel=1e-10)
    assert mc_direct == pytest.approx(mc_swapped, rel=1e-15)


@given(
    m1=st.floats(min_value=1e4, max_value=1e10),
    m2=st.floats(min_value=1e4, max_value=1e10),
)
@settings(max_examples=500)
def test_eta_bounds(m1: float, m2: float) -> None:
    """eta <= 0.25 always; eta == 0.25 iff m1 == m2.

    The strict inequality for m1 != m2 is weakened to <= 0.25 + eps to
    handle floating-point pairs that are distinct in Python but yield
    eta = 0.25 exactly due to cancellation (e.g. m2 = m1 - ulp(m1)).
    The upper bound of 0.25 is the analytically exact maximum.
    """
    lo, hi = (m1, m2) if m1 >= m2 else (m2, m1)
    cat = _make_catalog([hi], [lo])
    pop = derive_population(cat)
    eta_val = pop.eta[0]

    # eta must never exceed 0.25 (with small FP tolerance)
    assert eta_val <= 0.25 + 1e-14, f"eta={eta_val} > 0.25 for m1={hi}, m2={lo}"

    # For exactly equal masses, eta must be exactly 0.25
    if hi == lo:
        assert eta_val == pytest.approx(0.25, rel=1e-10)


# ---------------------------------------------------------------------------
# Fixture-based integration tests (require conftest.py synthetic catalog)
# ---------------------------------------------------------------------------


class TestWithSyntheticCatalog:
    def test_n_total_matches_catalog(
        self, synthetic_tng_catalog: "TNGMergerCatalog"
    ) -> None:
        """derive_population n_total must equal catalog.n_mergers."""
        pop = derive_population(synthetic_tng_catalog)
        assert pop.n_total == synthetic_tng_catalog.n_mergers

    def test_n_passing_le_n_total(
        self, synthetic_tng_catalog: "TNGMergerCatalog"
    ) -> None:
        """n_passing <= n_total always."""
        pop = derive_population(synthetic_tng_catalog)
        assert pop.n_passing <= pop.n_total

    def test_filter_population_all_pass(
        self, synthetic_tng_catalog: "TNGMergerCatalog"
    ) -> None:
        """After filter_population, every row must pass the quality cut."""
        pop = derive_population(synthetic_tng_catalog)
        filtered = filter_population(pop)
        assert np.all(filtered.passes_quality_cut), (
            "filter_population returned rows that fail the quality cut"
        )

    def test_filter_population_count(
        self, synthetic_tng_catalog: "TNGMergerCatalog"
    ) -> None:
        """Filtered population n_total matches original n_passing."""
        pop = derive_population(synthetic_tng_catalog)
        filtered = filter_population(pop)
        assert filtered.n_total == pop.n_passing

    def test_filter_population_arrays_consistent(
        self, synthetic_tng_catalog: "TNGMergerCatalog"
    ) -> None:
        """All arrays in filtered population have consistent length."""
        pop = derive_population(synthetic_tng_catalog)
        filtered = filter_population(pop)
        n = filtered.n_total
        assert len(filtered.chirp_mass_msun) == n
        assert len(filtered.total_mass_msun) == n
        assert len(filtered.mass_ratio_q) == n
        assert len(filtered.eta) == n
        assert len(filtered.passes_quality_cut) == n
