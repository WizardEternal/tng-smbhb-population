"""Multi-messenger gap plot and supporting figures for the TNG SMBHB population.

This module provides the flagship visualization for portfolio repo #2:
**figure #9** — the multi-messenger gap funnel plot — plus three supporting
figures (mass distribution, redshift-mass scatter, dual-survey comparison).

Per EXECUTION_PLAN.md §5.3 and locked decision L10, the gap plot is the single
most important figure in the entire three-repo portfolio.  It communicates in
ten seconds why most TNG progenitors fall in an observational gap: too heavy
for LISA, too light for PTAs, and with orbital periods outside optical survey
windows.

References
----------
Lin, Charisi & Haiman 2026, ApJ 997, 316
    LS recovery fractions applied to bars 6-7.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import numpy.typing as npt

from tng_smbhb.population import TNGPopulation
from tng_smbhb.gw_classification import GWClassification
from tng_smbhb.em_detectability import EMClassification

__all__ = [
    "FunnelStage",
    "compute_funnel_stages",
    "make_gap_plot",
    "make_mass_distribution_plot",
    "make_redshift_mass_plot",
    "make_gap_plot_dual_survey",
]

# ---------------------------------------------------------------------------
# FunnelStage dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunnelStage:
    """A single stage in the multi-messenger funnel.

    Attributes
    ----------
    label : str
        Human-readable stage label for y-axis display.
    count : float
        Number of systems at this stage (float to accommodate Lin+2026
        fractional expected counts for bars 6-7).
    is_expected : bool
        True for Lin+2026-fraction-weighted bars (stages 6 and 7); False for
        hard integer counts.
    """

    label: str
    count: float
    is_expected: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _theme_colors(theme: Literal["dark", "light"]) -> dict[str, str]:
    """Return a mapping of named colors for the chosen theme.

    Parameters
    ----------
    theme : {"dark", "light"}
        Visual theme.

    Returns
    -------
    dict[str, str]
        Keys: bg, fg, grid, annotation.
    """
    if theme == "dark":
        return {
            "bg": "#0a0a0a",
            "fg": "white",
            "grid": "#2a2a2a",
            "annotation": "#aaaaaa",
        }
    else:
        return {
            "bg": "white",
            "fg": "#111111",
            "grid": "#dddddd",
            "annotation": "#555555",
        }


def _apply_theme(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, theme: Literal["dark", "light"]) -> dict[str, str]:
    """Apply background/foreground theme colors to fig and ax.

    Parameters
    ----------
    fig : Figure
    ax : Axes
    theme : {"dark", "light"}

    Returns
    -------
    dict[str, str]
        The color map from _theme_colors.
    """
    colors = _theme_colors(theme)
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    ax.tick_params(colors=colors["fg"], which="both", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(colors["fg"])
    ax.xaxis.label.set_color(colors["fg"])
    ax.yaxis.label.set_color(colors["fg"])
    ax.title.set_color(colors["fg"])
    return colors


# ---------------------------------------------------------------------------
# Public: compute_funnel_stages  (no matplotlib needed)
# ---------------------------------------------------------------------------


def _compute_funnel_stages(
    pop: TNGPopulation,
    gwc: GWClassification,
    emc: EMClassification,
    survey: Literal["stripe82", "lsst"],
) -> list[FunnelStage]:
    """Internal implementation — called by the public wrapper."""
    q = pop.passes_quality_cut

    if survey == "stripe82":
        window_label = "P_obs in Stripe 82 window (200-1100 d)"
        in_survey = emc.in_stripe82
        exp_sin = emc.expected_sin_stripe82
        exp_saw = emc.expected_saw_stripe82
    else:
        window_label = "P_obs in LSST window (100-1200 d)"
        in_survey = emc.in_lsst
        exp_sin = emc.expected_sin_lsst
        exp_saw = emc.expected_saw_lsst

    stages: list[FunnelStage] = [
        FunnelStage(
            label="All TNG mergers",
            count=float(pop.n_total),
            is_expected=False,
        ),
        FunnelStage(
            label="Quality cut (M_tot > 1.2×10⁶ M☉)",
            count=float(pop.n_passing),
            is_expected=False,
        ),
        FunnelStage(
            label="PTA-band f_ISCO",
            count=float(np.sum(q & gwc.in_pta)),
            is_expected=False,
        ),
        FunnelStage(
            label="LISA-band f_ISCO",
            count=float(np.sum(q & gwc.in_lisa)),
            is_expected=False,
        ),
        FunnelStage(
            label=window_label,
            count=float(np.sum(q & in_survey)),
            is_expected=False,
        ),
        FunnelStage(
            label="Sinusoidal-recoverable (Lin+2026)",
            count=float(np.sum(q * exp_sin)),
            is_expected=True,
        ),
        FunnelStage(
            label="Sawtooth-recoverable (Lin+2026)",
            count=float(np.sum(q * exp_saw)),
            is_expected=True,
        ),
    ]
    return stages


def compute_funnel_stages(
    pop: TNGPopulation,
    gwc: GWClassification,
    emc: EMClassification,
    *,
    survey: Literal["stripe82", "lsst"] = "stripe82",
) -> list[FunnelStage]:
    """Compute the seven funnel stages for the multi-messenger gap plot.

    This public function can be imported and called by scripts/ and tests/
    to print text tables without importing matplotlib.

    Parameters
    ----------
    pop : TNGPopulation
        Full population from derive_population.
    gwc : GWClassification
        GW band classification from classify_bands.
    emc : EMClassification
        EM detectability classification from classify_em_detectability.
    survey : {"stripe82", "lsst"}
        Which survey to use for the window and recovery stages.

    Returns
    -------
    list[FunnelStage]
        Seven stages in order from widest (all mergers) to narrowest
        (sawtooth-recoverable).
    """
    return _compute_funnel_stages(pop, gwc, emc, survey)


# ---------------------------------------------------------------------------
# Internal: compute per-bar annotation strings
# ---------------------------------------------------------------------------


def _pct_string(count: float, prev: float | None) -> str:
    """Format a percentage string relative to *prev*.

    Parameters
    ----------
    count : float
        Current bar count.
    prev : float or None
        Previous non-zero bar count.  None or 0.0 → "--".

    Returns
    -------
    str
        E.g. "  (42.3% of prev)" or "  (--)"
    """
    if prev is None or prev == 0.0:
        return "  (--)"
    return f"  ({count / prev * 100:.1f}% of prev)"


def _format_count(count: float, is_expected: bool) -> str:
    """Format count as integer string or 1-decimal float string.

    Parameters
    ----------
    count : float
    is_expected : bool

    Returns
    -------
    str
    """
    if is_expected:
        return f"{count:,.1f}"
    return f"{int(round(count)):,d}"


# ---------------------------------------------------------------------------
# Public: make_gap_plot
# ---------------------------------------------------------------------------


def make_gap_plot(
    pop: TNGPopulation,
    gwc: GWClassification,
    emc: EMClassification,
    *,
    survey: Literal["stripe82", "lsst"] = "stripe82",
    outpath: str | Path,
    theme: Literal["dark", "light"] = "dark",
    title: str | None = None,
) -> Path:
    """Render THE multi-messenger gap plot (portfolio figure #9).

    Horizontal log-scale bar chart, seven stages from top to bottom:

      1. All TNG mergers
      2. + Quality cut (M_tot > 1.2e6 M_sun)
      3. PTA-band f_ISCO (roughly M_tot >= 1e8)
      4. LISA-band f_ISCO (roughly M_tot <= 1e7)
      5. P_obs in <survey> window (e.g. 200-1100 d for Stripe 82)
      6. Sinusoidal-recoverable (x Lin+2026 sinusoidal fraction)
      7. Sawtooth-recoverable (x Lin+2026 sawtooth fraction)

    Each bar annotated with absolute count and % of previous non-zero stage.
    Bottom caption: "Of N TNG progenitors, fewer than M are recoverable by
    Lomb-Scargle assuming sawtooth signals (Lin+2026)."

    Bars 6-7 are expected counts (fractional) from the aggregate Lin+2026
    recovery rates.

    Parameters
    ----------
    pop, gwc, emc : aligned dataclasses from upstream pipeline.
    survey : one of {"stripe82", "lsst"}.
    outpath : output path (PNG). Parent directory is created if missing.
    theme : "dark" (default) or "light".
    title : optional; a sensible default is used if None.

    Returns
    -------
    Path : the outpath that was written.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    stages = _compute_funnel_stages(pop, gwc, emc, survey)
    n_stages = len(stages)

    # --- Figure and axes ---
    if theme == "dark":
        plt.style.use("dark_background")
    else:
        matplotlib.rcdefaults()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = _apply_theme(fig, ax, theme)

    # --- Bar colors ---
    # Stages 0-4 (index 0..4): viridis gradient; stages 5-6: firebrick tones
    viridis = plt.get_cmap("viridis")
    n_gradient = 5  # stages 1-5
    gradient_colors = [viridis(0.85 - i * 0.15) for i in range(n_gradient)]
    expected_colors = ["firebrick", "darkred"]

    bar_colors: list[tuple] = gradient_colors + expected_colors

    # y-positions: stage 0 at top → y = 0, stage 6 at bottom → y = -6
    y_positions = [-i for i in range(n_stages)]

    # Track previous non-zero count for percentage calculation
    prev_count: float | None = None

    # Store bar lengths and annotation info
    bar_values: list[float] = []
    bar_annotations: list[str] = []

    for i, stage in enumerate(stages):
        bar_values.append(stage.count if stage.count > 0 else 1e-6)
        annotation = _format_count(stage.count, stage.is_expected) + _pct_string(
            stage.count, prev_count
        )
        bar_annotations.append(annotation)
        if stage.count > 0:
            prev_count = stage.count

    # Draw bars using barh (left-to-right along x)
    bar_height = 0.6
    bars = ax.barh(
        y_positions,
        bar_values,
        height=bar_height,
        color=bar_colors,
        edgecolor=colors["fg"],
        linewidth=0.5,
        align="center",
    )

    # --- Annotations at bar ends ---
    x_max = max(bar_values)
    for i, (bar, annotation, stage) in enumerate(zip(bars, bar_annotations, stages)):
        bar_width = bar.get_width()
        # Place text just to the right of the bar end
        ax.text(
            bar_width * 1.05,
            y_positions[i],
            annotation,
            va="center",
            ha="left",
            fontsize=8.5,
            color=colors["fg"],
        )

    # --- y-axis labels ---
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [s.label for s in stages],
        fontsize=10,
    )
    ax.tick_params(axis="y", colors=colors["fg"])

    # --- x-axis ---
    ax.set_xscale("log")
    ax.set_xlabel("Number of systems (log scale)", fontsize=12, color=colors["fg"])

    # Give room for annotations to the right
    ax.set_xlim(0.5, x_max * 30)

    # y-axis limits: a little padding above stage 0 and below stage 6
    ax.set_ylim(y_positions[-1] - 0.7, y_positions[0] + 0.7)

    # Hide left spine for cleanliness
    ax.spines["left"].set_visible(False)

    # --- Grid ---
    ax.grid(
        True,
        which="major",
        axis="x",
        color=colors["grid"],
        linestyle="--",
        linewidth=0.5,
        zorder=0,
    )

    # --- Legend for bar color groups ---
    patch_gradient = mpatches.Patch(
        color=viridis(0.7),
        label="Hard selection stages (1-5)",
    )
    patch_expected = mpatches.Patch(
        color="firebrick",
        label="Expected (Lin+2026 fraction × window)",
    )
    leg = ax.legend(
        handles=[patch_gradient, patch_expected],
        loc="lower right",
        fontsize=9,
        framealpha=0.4,
        facecolor=colors["bg"],
        edgecolor=colors["fg"],
        labelcolor=colors["fg"],
    )
    for text in leg.get_texts():
        text.set_color(colors["fg"])

    # --- Title ---
    if title is None:
        title = f"Multi-messenger gap: TNG SMBHB mergers → {survey.upper()} recoverable"
    ax.set_title(title, fontsize=13, color=colors["fg"], pad=14)

    # --- Bottom caption ---
    n_total = int(pop.n_total)
    sawtooth_stage = stages[-1]
    m_saw = sawtooth_stage.count
    caption = (
        f"Of {n_total:,d} TNG progenitors, fewer than {m_saw:,.1f} are recoverable by "
        f"Lomb-Scargle assuming sawtooth signals (Lin+2026)."
    )
    fig.text(
        0.5,
        0.01,
        caption,
        ha="center",
        va="bottom",
        fontsize=8,
        color=colors["annotation"],
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    fig.savefig(
        outpath,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    return outpath


# ---------------------------------------------------------------------------
# Public: make_mass_distribution_plot
# ---------------------------------------------------------------------------


def make_mass_distribution_plot(
    pop: TNGPopulation,
    *,
    outpath: str | Path,
    theme: Literal["dark", "light"] = "dark",
) -> Path:
    """Histogram of log10(total mass), overlaid with log10(chirp mass).

    Mark the quality-cut line at M_tot = 1.2e6 M_sun.

    Parameters
    ----------
    pop : TNGPopulation
    outpath : output path (PNG).
    theme : "dark" or "light".

    Returns
    -------
    Path : the outpath that was written.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if theme == "dark":
        plt.style.use("dark_background")
    else:
        matplotlib.rcdefaults()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = _apply_theme(fig, ax, theme)

    log_total = np.log10(pop.total_mass_msun)
    log_chirp = np.log10(pop.chirp_mass_msun)

    bins = np.linspace(
        min(log_total.min(), log_chirp.min()) - 0.2,
        max(log_total.max(), log_chirp.max()) + 0.2,
        50,
    )

    ax.hist(
        log_total,
        bins=bins,
        alpha=0.75,
        color="#4393c3",
        label=r"$\log_{10}(M_{\rm tot}/M_\odot)$",
        zorder=3,
    )
    ax.hist(
        log_chirp,
        bins=bins,
        alpha=0.6,
        color="#d6604d",
        label=r"$\log_{10}(\mathcal{M}/M_\odot)$",
        zorder=2,
    )

    # Quality cut line
    q_cut = np.log10(pop.quality_cut_min_total_mass_msun)
    ax.axvline(
        q_cut,
        color="gold",
        linewidth=1.8,
        linestyle="--",
        label=rf"Quality cut  ($M_{{\rm tot}} > 10^{{{q_cut:.1f}}}\,M_\odot$)",
        zorder=5,
    )

    ax.set_xlabel(r"$\log_{10}(M / M_\odot)$", fontsize=12, color=colors["fg"])
    ax.set_ylabel("Count", fontsize=12, color=colors["fg"])
    ax.set_title("TNG SMBHB merger mass distribution", fontsize=13, color=colors["fg"])

    leg = ax.legend(
        fontsize=9,
        framealpha=0.4,
        facecolor=colors["bg"],
        edgecolor=colors["fg"],
        labelcolor=colors["fg"],
    )
    for text in leg.get_texts():
        text.set_color(colors["fg"])

    ax.grid(
        True,
        which="major",
        color=colors["grid"],
        linestyle="--",
        linewidth=0.5,
        zorder=0,
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# Public: make_redshift_mass_plot
# ---------------------------------------------------------------------------

_BAND_PLOT_COLORS: dict[str, str] = {
    "pta": "teal",
    "lisa": "gold",
    "gap": "#e07b39",
    "neither": "#888888",
}


def make_redshift_mass_plot(
    pop: TNGPopulation,
    gwc: GWClassification,
    *,
    outpath: str | Path,
    theme: Literal["dark", "light"] = "dark",
) -> Path:
    """Scatter plot: redshift vs log10(M_tot), colored by GW band.

    Bands: PTA (teal), LISA (gold), gap (orange), neither (grey).

    Parameters
    ----------
    pop : TNGPopulation
    gwc : GWClassification
    outpath : output path (PNG).
    theme : "dark" or "light".

    Returns
    -------
    Path : the outpath that was written.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if theme == "dark":
        plt.style.use("dark_background")
    else:
        matplotlib.rcdefaults()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = _apply_theme(fig, ax, theme)

    log_m = np.log10(pop.total_mass_msun)
    z = pop.catalog.redshift

    in_neither = ~gwc.in_pta & ~gwc.in_lisa & ~gwc.in_gap

    band_masks: list[tuple[npt.NDArray[np.bool_], str, str]] = [
        (in_neither, "neither", "Neither"),
        (gwc.in_gap, "gap", "Gap (PTA-LISA, undetected)"),
        (gwc.in_lisa, "lisa", "LISA band"),
        (gwc.in_pta, "pta", "PTA band"),
    ]

    for mask, band_key, label in band_masks:
        if not np.any(mask):
            continue
        ax.scatter(
            z[mask],
            log_m[mask],
            c=_BAND_PLOT_COLORS[band_key],
            s=6,
            alpha=0.5,
            label=label,
            zorder=3 if band_key in ("gap", "lisa", "pta") else 2,
            rasterized=True,
        )

    ax.set_xlabel("Redshift  $z$", fontsize=12, color=colors["fg"])
    ax.set_ylabel(r"$\log_{10}(M_{\rm tot} / M_\odot)$", fontsize=12, color=colors["fg"])
    ax.set_title(
        "TNG SMBHB mergers: redshift vs total mass (colored by GW band)",
        fontsize=12,
        color=colors["fg"],
    )

    leg = ax.legend(
        fontsize=9,
        framealpha=0.4,
        facecolor=colors["bg"],
        edgecolor=colors["fg"],
        labelcolor=colors["fg"],
        markerscale=2.0,
    )
    for text in leg.get_texts():
        text.set_color(colors["fg"])

    ax.grid(
        True,
        which="major",
        color=colors["grid"],
        linestyle="--",
        linewidth=0.5,
        zorder=0,
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# Public: make_gap_plot_dual_survey
# ---------------------------------------------------------------------------


def make_gap_plot_dual_survey(
    pop: TNGPopulation,
    gwc: GWClassification,
    emc: EMClassification,
    *,
    outpath: str | Path,
    theme: Literal["dark", "light"] = "dark",
) -> Path:
    """2-panel side-by-side: Stripe 82 gap plot + LSST gap plot.

    For the 3 READMEs.  Each panel is a self-contained funnel following the
    same visual conventions as make_gap_plot.

    Parameters
    ----------
    pop : TNGPopulation
    gwc : GWClassification
    emc : EMClassification
    outpath : output path (PNG).
    theme : "dark" or "light".

    Returns
    -------
    Path : the outpath that was written.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if theme == "dark":
        plt.style.use("dark_background")
    else:
        matplotlib.rcdefaults()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    fig.patch.set_facecolor(_theme_colors(theme)["bg"])

    for ax, survey in zip(axes, ["stripe82", "lsst"]):
        _draw_gap_funnel_on_ax(ax, pop, gwc, emc, survey=survey, theme=theme)

    survey_label_s82 = "Stripe 82 (200-1100 d)"
    survey_label_lsst = "LSST (100-1200 d)"
    axes[0].set_title(
        f"Multi-messenger gap → {survey_label_s82}",
        fontsize=12,
        color=_theme_colors(theme)["fg"],
        pad=10,
    )
    axes[1].set_title(
        f"Multi-messenger gap → {survey_label_lsst}",
        fontsize=12,
        color=_theme_colors(theme)["fg"],
        pad=10,
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# Internal: shared funnel drawing on a given Axes
# ---------------------------------------------------------------------------


def _draw_gap_funnel_on_ax(
    ax: matplotlib.axes.Axes,
    pop: TNGPopulation,
    gwc: GWClassification,
    emc: EMClassification,
    *,
    survey: Literal["stripe82", "lsst"],
    theme: Literal["dark", "light"],
) -> None:
    """Draw the funnel bar chart directly on *ax* (shared by single + dual plots).

    Parameters
    ----------
    ax : Axes
    pop, gwc, emc : pipeline outputs.
    survey : which survey.
    theme : visual theme.
    """
    colors = _theme_colors(theme)
    ax.set_facecolor(colors["bg"])
    ax.tick_params(colors=colors["fg"], which="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(colors["fg"])
    ax.xaxis.label.set_color(colors["fg"])
    ax.yaxis.label.set_color(colors["fg"])

    stages = _compute_funnel_stages(pop, gwc, emc, survey)
    n_stages = len(stages)

    viridis = plt.get_cmap("viridis")
    gradient_colors = [viridis(0.85 - i * 0.15) for i in range(5)]
    expected_colors = ["firebrick", "darkred"]
    bar_colors_list: list[tuple] = gradient_colors + expected_colors

    y_positions = [-i for i in range(n_stages)]

    prev_count: float | None = None
    bar_values: list[float] = []
    bar_annotations: list[str] = []

    for stage in stages:
        bar_values.append(stage.count if stage.count > 0 else 1e-6)
        annotation = _format_count(stage.count, stage.is_expected) + _pct_string(
            stage.count, prev_count
        )
        bar_annotations.append(annotation)
        if stage.count > 0:
            prev_count = stage.count

    bar_height = 0.6
    bars = ax.barh(
        y_positions,
        bar_values,
        height=bar_height,
        color=bar_colors_list,
        edgecolor=colors["fg"],
        linewidth=0.5,
        align="center",
    )

    x_max = max(bar_values)
    for bar, annotation, y in zip(bars, bar_annotations, y_positions):
        ax.text(
            bar.get_width() * 1.05,
            y,
            annotation,
            va="center",
            ha="left",
            fontsize=7.5,
            color=colors["fg"],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([s.label for s in stages], fontsize=8.5)
    ax.tick_params(axis="y", colors=colors["fg"])
    ax.set_xscale("log")
    ax.set_xlabel("Number of systems (log scale)", fontsize=10, color=colors["fg"])
    ax.set_xlim(0.5, x_max * 30)
    ax.set_ylim(y_positions[-1] - 0.7, y_positions[0] + 0.7)
    ax.spines["left"].set_visible(False)
    ax.grid(
        True,
        which="major",
        axis="x",
        color=colors["grid"],
        linestyle="--",
        linewidth=0.4,
        zorder=0,
    )
