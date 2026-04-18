"""Tests for plotting.compute_funnel_stages and file-render smoke checks.

Rendering tests use the Agg backend and write to tmp_path, so they are
headless-safe for CI.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tng_smbhb.catalog import catalog_from_arrays
from tng_smbhb.em_detectability import classify_em_detectability
from tng_smbhb.gw_classification import classify_bands
from tng_smbhb.plotting import (
    FunnelStage,
    compute_funnel_stages,
    make_gap_plot,
    make_gap_plot_dual_survey,
    make_mass_distribution_plot,
    make_redshift_mass_plot,
)
from tng_smbhb.population import derive_population


# ---------------------------------------------------------------------------
# Fixtures local to this file
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _matplotlib_teardown():
    """Release every matplotlib figure after each test.

    The plotting helpers already call ``plt.close(fig)`` internally, but
    matplotlib also retains font/renderer state in ``pyplot`` module-level
    caches.  Under Windows + Python 3.12 this accumulates across many
    PNG renders in a single pytest process and intermittently blows up
    with ``MemoryError: bad allocation`` in the Agg backend's preview
    renderer.  ``plt.close("all")`` after each test is the standard
    matplotlib-test hygiene to prevent that.
    """
    yield
    plt.close("all")


@pytest.fixture
def pipeline_results(synthetic_tng_catalog):
    """Run the upstream pipeline once and reuse in every test."""
    pop = derive_population(synthetic_tng_catalog)
    gwc = classify_bands(pop)
    emc = classify_em_detectability(pop, gwc.f_isco_hz)
    return pop, gwc, emc


# ---------------------------------------------------------------------------
# compute_funnel_stages
# ---------------------------------------------------------------------------


def test_funnel_has_seven_stages(pipeline_results) -> None:
    pop, gwc, emc = pipeline_results
    stages = compute_funnel_stages(pop, gwc, emc, survey="stripe82")
    assert len(stages) == 7
    for stage in stages:
        assert isinstance(stage, FunnelStage)
        assert stage.label
        assert stage.count >= 0.0


def test_funnel_stripe82_first_stage_is_total(pipeline_results) -> None:
    pop, gwc, emc = pipeline_results
    stages = compute_funnel_stages(pop, gwc, emc, survey="stripe82")
    assert stages[0].count == pop.n_total


def test_funnel_last_two_stages_are_expected(pipeline_results) -> None:
    pop, gwc, emc = pipeline_results
    stages = compute_funnel_stages(pop, gwc, emc, survey="stripe82")
    # Bars 5 (sinusoidal-recoverable) and 6 (sawtooth-recoverable) are
    # fraction-weighted expected counts from Lin+2026.
    assert stages[5].is_expected
    assert stages[6].is_expected
    # Sawtooth fraction (0.09) <= sinusoidal fraction (0.45).
    assert stages[6].count <= stages[5].count + 1e-9


def test_funnel_lsst_variant(pipeline_results) -> None:
    pop, gwc, emc = pipeline_results
    stages = compute_funnel_stages(pop, gwc, emc, survey="lsst")
    assert len(stages) == 7
    assert stages[0].count == pop.n_total


def test_funnel_counts_consistent_with_masks(pipeline_results) -> None:
    """The PTA/LISA-band bar counts must equal the population masks AND quality."""
    pop, gwc, emc = pipeline_results
    stages = compute_funnel_stages(pop, gwc, emc, survey="stripe82")
    n_quality_pta = int(np.sum(pop.passes_quality_cut & gwc.in_pta))
    n_quality_lisa = int(np.sum(pop.passes_quality_cut & gwc.in_lisa))
    assert stages[2].count == float(n_quality_pta)
    assert stages[3].count == float(n_quality_lisa)


# ---------------------------------------------------------------------------
# File-render smoke tests
# ---------------------------------------------------------------------------


def test_gap_plot_renders(pipeline_results, tmp_path) -> None:
    pop, gwc, emc = pipeline_results
    out = tmp_path / "gap.png"
    returned = make_gap_plot(
        pop, gwc, emc, survey="stripe82", outpath=out, theme="dark"
    )
    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 5_000  # non-trivial PNG


def test_gap_plot_light_theme_renders(pipeline_results, tmp_path) -> None:
    pop, gwc, emc = pipeline_results
    out = tmp_path / "gap_light.png"
    make_gap_plot(pop, gwc, emc, survey="stripe82", outpath=out, theme="light")
    assert out.stat().st_size > 5_000


def test_gap_plot_lsst_renders(pipeline_results, tmp_path) -> None:
    pop, gwc, emc = pipeline_results
    out = tmp_path / "gap_lsst.png"
    make_gap_plot(pop, gwc, emc, survey="lsst", outpath=out)
    assert out.stat().st_size > 5_000


def test_gap_plot_dual_survey_renders(pipeline_results, tmp_path) -> None:
    pop, gwc, emc = pipeline_results
    out = tmp_path / "gap_dual.png"
    make_gap_plot_dual_survey(pop, gwc, emc, outpath=out)
    assert out.stat().st_size > 5_000


def test_mass_distribution_renders(pipeline_results, tmp_path) -> None:
    pop, _gwc, _emc = pipeline_results
    out = tmp_path / "mass_dist.png"
    make_mass_distribution_plot(pop, outpath=out)
    assert out.stat().st_size > 5_000


def test_redshift_mass_renders(pipeline_results, tmp_path) -> None:
    pop, gwc, _emc = pipeline_results
    out = tmp_path / "z_mass.png"
    make_redshift_mass_plot(pop, gwc, outpath=out)
    assert out.stat().st_size > 5_000


def test_gap_plot_creates_parent_dir(pipeline_results, tmp_path) -> None:
    """Ensure the plotting helper creates missing parent directories."""
    pop, gwc, emc = pipeline_results
    out = tmp_path / "nested" / "subdir" / "gap.png"
    make_gap_plot(pop, gwc, emc, survey="stripe82", outpath=out)
    assert out.exists()
