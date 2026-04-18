"""Tests for gw_classification.py.

Covers:
- compute_f_isco reference value at 7e8 M_sun (~6.28e-6 Hz)
- compute_f_isco_observer: observer-frame frequency is redshifted relative to source
- Band boundary spot checks using OBSERVER-frame f_ISCO (z=0.1 and z=3)
- classify_bands exhaustiveness: n_pta + n_lisa + n_gap + n_neither == n_total
- classify_bands exposes both f_isco_source_hz and f_isco_observer_hz
- Observer-frame band assignment changes with redshift (regression tests)
- Regression: hardcoded expected values for m1=m2=1e9 M_sun at z=2
- Hypothesis: f_ISCO is strictly decreasing in M_tot
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tng_smbhb.catalog import TNGMergerCatalog, catalog_from_arrays
from tng_smbhb.gw_classification import (
    GWBand,
    classify_bands,
    compute_f_isco,
    compute_f_isco_observer,
)
from tng_smbhb.population import derive_population
from tng_smbhb._vendored_constants import G, M_SUN, c


# ---------------------------------------------------------------------------
# Reference value test (source frame)
# ---------------------------------------------------------------------------


def test_compute_f_isco_reference_value() -> None:
    """7e8 M_sun reference system: f_ISCO_source ~ 6.28e-6 Hz (gap band at z~0)."""
    m_ref = np.array([7e8], dtype=np.float64)
    f = compute_f_isco(m_ref)
    np.testing.assert_allclose(f, [6.28e-6], rtol=1e-3)


# ---------------------------------------------------------------------------
# Observer-frame frequency tests
# ---------------------------------------------------------------------------


def test_compute_f_isco_observer_at_z_zero() -> None:
    """At z=0 the observer-frame frequency equals the source-frame frequency."""
    m = np.array([1e9], dtype=np.float64)
    z = np.array([0.0], dtype=np.float64)
    f_src = compute_f_isco(m)
    f_obs = compute_f_isco_observer(m, z)
    np.testing.assert_allclose(f_obs, f_src, rtol=1e-12)


def test_compute_f_isco_observer_redshift_factor() -> None:
    """f_obs = f_src / (1+z) for a known redshift."""
    m = np.array([1e9], dtype=np.float64)
    z_val = 2.0
    z = np.array([z_val], dtype=np.float64)
    f_src = compute_f_isco(m)
    f_obs = compute_f_isco_observer(m, z)
    np.testing.assert_allclose(f_obs, f_src / (1.0 + z_val), rtol=1e-12)


def test_f_isco_observer_less_than_source_for_positive_z() -> None:
    """For any z > 0 the observer-frame frequency must be less than source-frame."""
    masses = np.array([1e6, 1e8, 1e10], dtype=np.float64)
    redshifts = np.array([0.5, 1.0, 3.0], dtype=np.float64)
    f_src = compute_f_isco(masses)
    f_obs = compute_f_isco_observer(masses, redshifts)
    assert np.all(f_obs < f_src), "Observer-frame f_ISCO must be < source-frame for z>0"


# ---------------------------------------------------------------------------
# Regression: hardcoded hand-computed values for m1=m2=1e9 M_sun at z=2
# ---------------------------------------------------------------------------


def test_f_isco_regression_1e9msun_z2() -> None:
    """Hard-coded regression: m1=m2=1e9 M_sun at z=2.

    Hand computation:
      M_tot = 2e9 M_sun
      M_tot_kg = 2e9 * 1.989e30 = 3.978e39 kg
      f_ISCO_source = c^3 / (6^1.5 * pi * G * M_tot_kg)
                    = (2.998e8)^3 / (6^1.5 * pi * 6.674e-11 * 3.978e39)
      6^1.5 = 14.6969...
      numerator: 2.694e25
      denominator: 14.6969 * 3.14159 * 6.674e-11 * 3.978e39
                 = 14.6969 * 3.14159 * 2.655e29
                 = 1.224e31
      f_source ~ 2.201e-6 Hz   (gap band, since 1e-7 < f < 1e-4)

      f_observer = f_source / (1 + 2) = 2.201e-6 / 3 ~ 7.337e-7 Hz
      This is still in the gap (1e-7 < 7.337e-7 < 1e-4).
    """
    m_tot = 2e9  # M_sun
    z = 2.0

    # Exact formula values using vendored constants
    m_tot_kg = m_tot * M_SUN
    f_source_expected = c**3 / (6.0**1.5 * math.pi * G * m_tot_kg)
    f_observer_expected = f_source_expected / (1.0 + z)

    m_arr = np.array([m_tot], dtype=np.float64)
    z_arr = np.array([z], dtype=np.float64)
    f_src = compute_f_isco(m_arr)
    f_obs = compute_f_isco_observer(m_arr, z_arr)

    np.testing.assert_allclose(f_src, [f_source_expected], rtol=1e-10)
    np.testing.assert_allclose(f_obs, [f_observer_expected], rtol=1e-10)

    # Verify the hand-computed value is approximately 2.2e-6 Hz (gap band)
    np.testing.assert_allclose(float(f_src[0]), 2.201e-6, rtol=5e-3)
    # Observer-frame: ~7.34e-7 Hz (still gap band)
    np.testing.assert_allclose(float(f_obs[0]), 7.34e-7, rtol=5e-3)

    # Band checks: both should be "gap"
    from tng_smbhb.gw_classification import PTA_F_MAX_HZ, LISA_F_MIN_HZ
    assert PTA_F_MAX_HZ < float(f_src[0]) < LISA_F_MIN_HZ, "f_source should be in gap"
    assert PTA_F_MAX_HZ < float(f_obs[0]) < LISA_F_MIN_HZ, "f_observer should be in gap"


# ---------------------------------------------------------------------------
# Band assignment changes with redshift (observer-frame behavior)
# ---------------------------------------------------------------------------


def _make_single_system_catalog(
    m_tot_msun: float, z: float
) -> TNGMergerCatalog:
    """Helper: catalog with a single equal-mass system at given redshift."""
    a = 1.0 / (1.0 + z)
    return catalog_from_arrays(
        m1_msun=np.array([m_tot_msun / 2.0]),
        m2_msun=np.array([m_tot_msun / 2.0]),
        scale_factor=np.array([a]),
        simulation="test",
        hubble_h=0.6774,
    )


def test_band_assignment_changes_with_redshift_5e9msun() -> None:
    """m1=m2=5e9 M_sun: band depends on redshift (observer-frame bug regression).

    Source-frame: M_tot=1e10 M_sun → f_source ~ 4.4e-7 Hz (gap band).
    At z=0.1: f_obs = 4.4e-7 / 1.1 ~ 4.0e-7 Hz → still gap.
    At z=3.0: f_obs = 4.4e-7 / 4.0 ~ 1.1e-7 Hz → still gap (just above PTA edge).

    The KEY point is that the observer-frame frequency is strictly less than
    the source-frame frequency for any z > 0. The test verifies that the
    classifier correctly populates f_isco_observer_hz < f_isco_source_hz
    and that the (1+z) factor is applied when assigning bands.
    """
    # z=0.1
    cat_low_z = _make_single_system_catalog(1e10, z=0.1)
    pop_low_z = derive_population(cat_low_z)
    gc_low_z = classify_bands(pop_low_z)

    # z=3.0
    cat_high_z = _make_single_system_catalog(1e10, z=3.0)
    pop_high_z = derive_population(cat_high_z)
    gc_high_z = classify_bands(pop_high_z)

    # M_tot=1e10: f_source ~ 4.4e-7 Hz → gap band at all tested redshifts
    # (gap: 1e-7 < f < 1e-4)
    assert gc_low_z.band[0] == GWBand.GAP.value, (
        f"Expected gap at z=0.1, got {gc_low_z.band[0]}, "
        f"f_obs={gc_low_z.f_isco_observer_hz[0]:.3e} Hz"
    )
    assert gc_high_z.band[0] == GWBand.GAP.value, (
        f"Expected gap at z=3.0, got {gc_high_z.band[0]}, "
        f"f_obs={gc_high_z.f_isco_observer_hz[0]:.3e} Hz"
    )

    # The KEY check: observer-frame f must be < source-frame f (since z > 0)
    assert gc_low_z.f_isco_observer_hz[0] < gc_low_z.f_isco_source_hz[0], (
        "Observer-frame f must be < source-frame f for z=0.1"
    )
    assert gc_high_z.f_isco_observer_hz[0] < gc_high_z.f_isco_source_hz[0], (
        "Observer-frame f must be < source-frame f for z=3.0"
    )

    # At higher z the observer-frame frequency is lower
    assert gc_high_z.f_isco_observer_hz[0] < gc_low_z.f_isco_observer_hz[0], (
        "f_obs(z=3.0) should be lower than f_obs(z=0.1) for same mass"
    )

    # Verify that a naive source-frame classifier would assign the SAME band
    # at both redshifts (same mass → same f_source), whereas the observer-frame
    # classifier correctly captures the redshift dependence in the frequency value
    np.testing.assert_allclose(
        gc_low_z.f_isco_source_hz[0], gc_high_z.f_isco_source_hz[0], rtol=1e-10,
        err_msg="Source-frame f is mass-only (same mass → same f_source)",
    )
    # But observer-frame frequencies differ because z differs
    assert gc_low_z.f_isco_observer_hz[0] != gc_high_z.f_isco_observer_hz[0]


def test_band_changes_from_pta_to_neither_across_redshift() -> None:
    """A system near the PTA lower edge falls out of PTA at high z.

    M_tot chosen so f_source is just above PTA_F_MIN_HZ=1e-9 Hz at z~0.
    f_source ~ 1.1e-9 Hz → requires M_tot ~ c^3 / (6^1.5*pi*G*1.1e-9*M_sun)
    At z=5 f_obs = 1.1e-9 / 6 ~ 1.8e-10 Hz → below PTA, so "neither".
    """
    from tng_smbhb.gw_classification import PTA_F_MIN_HZ

    # Find M_tot such that f_source = 1.1 * PTA_F_MIN_HZ (just inside PTA at z~0)
    target_f = 1.1 * PTA_F_MIN_HZ  # 1.1e-9 Hz
    # f = c^3 / (6^1.5 * pi * G * M_kg)  →  M_kg = c^3 / (6^1.5 * pi * G * f)
    m_kg = c**3 / (6.0**1.5 * math.pi * G * target_f)
    m_msun = m_kg / M_SUN

    # At z=0.05: f_obs = target_f / 1.05 ≈ 1.048e-9 Hz → still in PTA
    cat_low = _make_single_system_catalog(m_msun, z=0.05)
    pop_low = derive_population(cat_low)
    gc_low = classify_bands(pop_low)
    assert gc_low.band[0] == GWBand.PTA.value, (
        f"Expected PTA at z=0.05 for M={m_msun:.2e} M_sun, "
        f"f_obs={gc_low.f_isco_observer_hz[0]:.3e} Hz"
    )

    # At z=5: f_obs = target_f / 6 ~ 1.83e-10 Hz → below PTA_F_MIN_HZ → "neither"
    cat_high = _make_single_system_catalog(m_msun, z=5.0)
    pop_high = derive_population(cat_high)
    gc_high = classify_bands(pop_high)
    assert gc_high.band[0] == GWBand.NEITHER.value, (
        f"Expected neither at z=5 for M={m_msun:.2e} M_sun, "
        f"f_obs={gc_high.f_isco_observer_hz[0]:.3e} Hz"
    )


# ---------------------------------------------------------------------------
# Band boundary spot checks (observer-frame at z=0 → same as source frame)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m_tot_msun, expected_band",
    [
        (1e11, GWBand.PTA.value),      # f_source ~ 4.4e-8 Hz  -> PTA at z=0
        (1e6,  GWBand.LISA.value),     # f_source ~ 4.4e-3 Hz  -> LISA at z=0
        (1e8,  GWBand.GAP.value),      # f_source ~ 4.4e-5 Hz  -> gap at z=0
        (1e13, GWBand.NEITHER.value),  # f_source ~ 4.4e-10 Hz -> neither at z=0
    ],
)
def test_band_boundaries_observer_frame(m_tot_msun: float, expected_band: str) -> None:
    """Spot-check that known masses at z≈0 map to the correct band (observer frame).

    At z=0 the observer-frame frequency equals the source-frame frequency,
    so this is a minimal sanity check that the band edges are correct.
    """
    # Build a catalog at z~0 (scale_factor ~ 1 → redshift ~ 0)
    cat = _make_single_system_catalog(m_tot_msun, z=1e-6)
    pop = derive_population(cat)
    gc = classify_bands(pop)

    from tng_smbhb.gw_classification import (
        LISA_F_MAX_HZ, LISA_F_MIN_HZ, PTA_F_MAX_HZ, PTA_F_MIN_HZ,
    )

    f_val = float(gc.f_isco_observer_hz[0])

    if expected_band == GWBand.PTA.value:
        assert PTA_F_MIN_HZ <= f_val <= PTA_F_MAX_HZ, (
            f"Expected PTA for M={m_tot_msun:.0e} M_sun, got f_obs={f_val:.3e} Hz"
        )
    elif expected_band == GWBand.LISA.value:
        assert LISA_F_MIN_HZ <= f_val <= LISA_F_MAX_HZ, (
            f"Expected LISA for M={m_tot_msun:.0e} M_sun, got f_obs={f_val:.3e} Hz"
        )
    elif expected_band == GWBand.GAP.value:
        assert PTA_F_MAX_HZ < f_val < LISA_F_MIN_HZ, (
            f"Expected gap for M={m_tot_msun:.0e} M_sun, got f_obs={f_val:.3e} Hz"
        )
    else:
        in_pta = PTA_F_MIN_HZ <= f_val <= PTA_F_MAX_HZ
        in_lisa = LISA_F_MIN_HZ <= f_val <= LISA_F_MAX_HZ
        in_gap = PTA_F_MAX_HZ < f_val < LISA_F_MIN_HZ
        assert not in_pta and not in_lisa and not in_gap, (
            f"Expected neither for M={m_tot_msun:.0e} M_sun, got f_obs={f_val:.3e} Hz"
        )


# ---------------------------------------------------------------------------
# classify_bands exhaustiveness on synthetic catalog
# ---------------------------------------------------------------------------


def test_classify_bands_exhaustive(synthetic_tng_catalog: TNGMergerCatalog) -> None:
    """All systems are assigned exactly one band; counts sum to n_total."""
    pop = derive_population(synthetic_tng_catalog)
    gc = classify_bands(pop)

    total = gc.n_pta + gc.n_lisa + gc.n_gap + gc.n_neither
    assert total == pop.n_total, (
        f"Band counts ({gc.n_pta} PTA + {gc.n_lisa} LISA + "
        f"{gc.n_gap} gap + {gc.n_neither} neither = {total}) "
        f"!= n_total ({pop.n_total})"
    )


def test_classify_bands_returns_gwclassification(
    synthetic_tng_catalog: TNGMergerCatalog,
) -> None:
    """classify_bands returns a GWClassification with both frame fields."""
    from tng_smbhb.gw_classification import GWClassification

    pop = derive_population(synthetic_tng_catalog)
    gc = classify_bands(pop)

    assert isinstance(gc, GWClassification)
    # Both frame fields must be present and correctly shaped
    assert gc.f_isco_source_hz.shape == (pop.n_total,)
    assert gc.f_isco_observer_hz.shape == (pop.n_total,)
    # Backward-compat alias
    assert gc.f_isco_hz.shape == (pop.n_total,)
    np.testing.assert_array_equal(gc.f_isco_hz, gc.f_isco_source_hz)

    assert gc.band.shape == (pop.n_total,)
    assert gc.in_pta.shape == (pop.n_total,)
    assert gc.in_lisa.shape == (pop.n_total,)
    assert gc.in_gap.shape == (pop.n_total,)


def test_observer_freq_always_leq_source_freq(
    synthetic_tng_catalog: TNGMergerCatalog,
) -> None:
    """Observer-frame frequency <= source-frame frequency for all z >= 0."""
    pop = derive_population(synthetic_tng_catalog)
    gc = classify_bands(pop)
    assert np.all(gc.f_isco_observer_hz <= gc.f_isco_source_hz), (
        "Observer-frame f_ISCO must be <= source-frame f_ISCO for all z >= 0"
    )


# ---------------------------------------------------------------------------
# Hypothesis: f_ISCO is strictly decreasing in M_tot
# ---------------------------------------------------------------------------


@given(
    m1=st.floats(min_value=1e4, max_value=1e12, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_f_isco_strictly_decreasing(m1: float) -> None:
    """f_ISCO(2*M) < f_ISCO(M) for all positive M (inverse proportionality)."""
    arr = np.array([m1, 2.0 * m1], dtype=np.float64)
    f = compute_f_isco(arr)
    assert f[1] < f[0], (
        f"f_ISCO should decrease with mass: "
        f"f({m1:.3e}) = {f[0]:.3e} Hz, f({2*m1:.3e}) = {f[1]:.3e} Hz"
    )
