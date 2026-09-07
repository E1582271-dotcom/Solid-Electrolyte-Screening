"""Render five main figures and one supplementary figure from frozen source tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from .config import (
    FIGURES,
    MAIN_FIGURES,
    METRICS,
    MODEL_SPECS,
    SOURCE_DATA,
    SUPPLEMENTARY,
    SUPPLEMENTARY_FIGURES,
)

FIGURE_WIDTH_IN = 7.2
MAX_FIGURE_HEIGHT_IN = 170 / 25.4
PANEL_LABEL_SIZE = 8.0

BLUE = "#3B6FB6"
SKY = "#8CBED6"
TEAL = "#2A9D8F"
RED = "#C65D21"
PURPLE = "#B07AA1"
GRAY = "#747A80"
MID_GRAY = "#8F969C"
LIGHT = "#D9DEE2"
INK = "#222222"
WHITE = "#FFFFFF"
OFF_WHITE = "#F7F7F5"
COLORBAR_FRACTION = 0.047
COLORBAR_PAD = 0.028
HEATMAP_LIMIT = 4.0
BUBBLE_MIN_AREA = 10.0
BUBBLE_SCALE_AREA = 50.0

LOG_SIGMA_LABEL = r"$\log_{10}[\sigma/(\mathrm{S\,cm}^{-1})]$"
MAE_LABEL = r"MAE (log$_{10}$ units)"
SPREAD_LABEL = r"Ensemble spread (log$_{10}$ units)"
ABSOLUTE_ERROR_LABEL = r"Absolute test error (log$_{10}$ units)"
RANKING_SCORE_LABEL = f"Ranking score, {LOG_SIGMA_LABEL}"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "nature_blue_ivory_ochre",
    ["#4F6D8A", "#F6F3EA", "#B17A3A"],
)
DISPOSITION_COLORS = {
    "reference_measurement": "#BCC2C7",
    "oxygen_scope_exclusion": MID_GRAY,
    "redox_electronic_risk": RED,
    "obelix_composition_reference": PURPLE,
    "screen_candidate": TEAL,
}
DISPOSITION_MARKERS = {
    "oxygen_scope_exclusion": "s",
    "redox_electronic_risk": "^",
    "obelix_composition_reference": "D",
    "screen_candidate": "o",
}
STATE_COLORS = {
    "extrapolative": RED,
    "high_score_lower_risk": TEAL,
    "unflagged": MID_GRAY,
}
STATE_MARKERS = {
    "extrapolative": "^",
    "high_score_lower_risk": "o",
    "unflagged": "s",
}

FIGURE_1_PALETTE = {
    "train_query": "#3F6D9E",
    "test_risk": "#9B2D3F",
    "oxygen_exclusion": "#B4B7BA",
    "risk_exclusion": "#9BA0A5",
    "obelix_reference": "#83878C",
    "candidate": "#173F6B",
}
FIGURE_1_MULTIPLICITY_COLORS = ("#8CAFD1", "#4A72A0", "#173F6B")


def _style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.2,
        "axes.labelsize": 6.4,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 5.6,
        "ytick.labelsize": 5.6,
        "legend.fontsize": 5.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.3,
        "ytick.major.size": 2.3,
        "legend.frameon": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "axes.grid": False,
        "savefig.facecolor": WHITE,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "mathtext.fontset": "stixsans",
        "pdf.fonttype": 42,
    })


def _panel(
    ax,
    label: str,
    title: str | None = None,
    *,
    title_x: float = 0.0,
) -> None:
    ax.text(
        -0.11,
        1.065,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )
    if title:
        ax.set_title(
            title,
            loc="left",
            x=title_x,
            pad=4,
            fontweight="bold",
            color=INK,
        )


def _tidy(ax) -> None:
    ax.spines[["top", "right"]].set_visible(True)
    ax.grid(False)
    ax.tick_params(direction="out", top=False, right=False)


def _align_columns(fig: plt.Figure, *columns) -> None:
    """Left-align vertically stacked axes after freezing the layout engine."""
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for axes in columns:
        x0 = max(axis.get_position().x0 for axis in axes)
        for axis in axes:
            position = axis.get_position()
            axis.set_position([x0, position.y0, position.x1 - x0, position.height])


def _align_rows(fig: plt.Figure, *rows) -> None:
    """Equalize the frame top/bottom of axes sharing a row after freezing layout."""
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for axes in rows:
        y1 = min(axis.get_position().y1 for axis in axes)
        y0 = max(axis.get_position().y0 for axis in axes)
        for axis in axes:
            position = axis.get_position()
            axis.set_position([position.x0, y0, position.width, y1 - y0])


def _match_colorbar(colorbar: mpl.colorbar.Colorbar, parent) -> None:
    """Re-seat a colorbar's vertical extent against its resized parent axes."""
    parent_position = parent.get_position()
    bar_position = colorbar.ax.get_position()
    colorbar.ax.set_position([
        bar_position.x0,
        parent_position.y0,
        bar_position.width,
        parent_position.height,
    ])


def _figure(height: float) -> plt.Figure:
    if height > MAX_FIGURE_HEIGHT_IN:
        raise ValueError("Nature figure height exceeds 170 mm")
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_IN, height),
        constrained_layout=True,
        facecolor=WHITE,
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(
            w_pad=0.035,
            h_pad=0.035,
            wspace=0.035,
            hspace=0.045,
        )
    return figure


def _colorbar(
    fig: plt.Figure,
    mappable,
    ax: plt.Axes,
    label: str,
    *,
    extend: str = "neither",
) -> mpl.colorbar.Colorbar:
    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=COLORBAR_FRACTION,
        pad=COLORBAR_PAD,
        extend=extend,
    )
    colorbar.set_label(label, labelpad=4)
    colorbar.ax.tick_params(width=0.6, length=2.3)
    colorbar.outline.set_linewidth(0.7)
    return colorbar


def _read_csv(name: str) -> pd.DataFrame:
    path = SOURCE_DATA / name
    if not path.exists():
        raise FileNotFoundError(f"missing figure source data: {path}")
    return pd.read_csv(path)


def _metrics() -> dict:
    return json.loads(METRICS.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, filename: str, directory: Path = FIGURES) -> None:
    if Path(filename).suffix.lower() != ".png":
        raise ValueError(f"figure output must be a 900-dpi PNG: {filename}")
    destination = directory / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.stem}.tmp.png"
    fig.savefig(
        temporary,
        dpi=900,
        facecolor=WHITE,
        edgecolor=WHITE,
        transparent=False,
        metadata={"Software": "p1screen 0.1.0; Matplotlib"},
    )
    plt.close(fig)
    with Image.open(temporary) as image:
        rgb = Image.new("RGB", image.size, WHITE)
        if image.mode == "RGBA":
            rgb.paste(image, mask=image.getchannel("A"))
        else:
            rgb.paste(image.convert("RGB"))
        rgb.save(destination, format="PNG", dpi=(900, 900), compress_level=6)
    temporary.unlink()


def _clean_figure_directory() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    keep = set(MAIN_FIGURES)
    for path in FIGURES.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()


def _clean_supplementary_directory() -> None:
    SUPPLEMENTARY.mkdir(parents=True, exist_ok=True)
    keep = set(SUPPLEMENTARY_FIGURES)
    for path in SUPPLEMENTARY.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()


def figure_01() -> None:
    audit = _read_csv("fig01b_split_audit.csv").iloc[0]
    disposition = _read_csv("fig01c_mp_disposition.csv")
    query_contract = _read_csv("fig01c_mp_query_contract.csv").iloc[0]
    queue = _read_csv("fig05_candidate_queue.csv")
    n_total = int(audit.n_train_original + audit.n_test)
    n_compositions = int(
        audit.n_train_composition_groups
        + audit.n_test_composition_groups
        - audit.n_composition_overlap_groups
    )
    n_censored = int(audit.n_censored_train + audit.n_censored_test)
    fig = _figure(5.35)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.84, 1.16])
    top = outer[0].subgridspec(1, 2, width_ratios=[0.85, 1.35], wspace=0.16)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.34, 0.86], wspace=0.18)

    ax = fig.add_subplot(top[0, 0])
    _panel(ax, "a", "Measured evidence")
    values = [int(audit.n_train_original), int(audit.n_test)]
    bars = ax.bar(
        [0, 1],
        values,
        width=0.56,
        color=[FIGURE_1_PALETTE["train_query"], FIGURE_1_PALETTE["test_risk"]],
        zorder=3,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 9,
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.text(
        0.98,
        0.94,
        f"{n_total} records\n{n_compositions} compositions\n{n_censored} upper bounds",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        linespacing=1.3,
    )
    ax.set_xticks([0, 1], ["Official train", "Held-out test"])
    ax.set_ylabel("Records (count)")
    ax.set_ylim(0, 550)
    _tidy(ax)
    ax_measured = ax

    ax = fig.add_subplot(top[0, 1])
    _panel(ax, "b", "Leakage-corrected split")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.42,rounding_size=0.10", fc=WHITE, lw=1.0)
    ax.text(
        2.0,
        4.12,
        f"Official train\n{int(audit.n_train_original)} records · "
        f"{int(audit.n_train_composition_groups)} groups",
        ha="center",
        va="center",
        bbox={**box, "ec": FIGURE_1_PALETTE["train_query"]},
    )
    ax.text(
        8.0,
        4.12,
        f"Official test\n{int(audit.n_test)} records · "
        f"{int(audit.n_test_composition_groups)} groups",
        ha="center",
        va="center",
        bbox={**box, "ec": FIGURE_1_PALETTE["test_risk"]},
    )
    for start, end in [((2.75, 3.58), (4.18, 3.04)), ((7.25, 3.58), (5.82, 3.04))]:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.9},
        )
    ax.text(
        5,
        2.80,
        f"{int(audit.n_composition_overlap_groups)} shared compositions",
        ha="center",
        va="center",
        fontweight="bold",
        bbox={**box, "ec": FIGURE_1_PALETTE["test_risk"]},
    )
    ax.annotate(
        "",
        xy=(5, 1.70),
        xytext=(5, 2.28),
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.9},
    )
    ax.text(
        5,
        1.22,
        f"Strict protocol · remove "
        f"{int(audit.n_train_rows_removed_for_strict_test)} train rows\n"
        f"{int(audit.n_train_strict)} train | {int(audit.n_test)} test",
        ha="center",
        va="center",
        fontweight="bold",
        bbox={**box, "ec": FIGURE_1_PALETTE["train_query"]},
    )
    ax.text(
        5,
        0.12,
        f"DOI overlap {int(audit.n_doi_overlap)} · composition-novel test "
        f"{int(audit.n_test_composition_novel)}",
        ha="center",
        color=GRAY,
    )

    ax = fig.add_subplot(bottom[0, 0])
    _panel(ax, "c", "Frozen MP query and disposition")
    order = [
        "oxygen_scope_exclusion",
        "redox_electronic_risk",
        "obelix_composition_reference",
        "screen_candidate",
    ]
    labels = [
        "O-containing\nout of domain",
        "Redox/electronic\nrisk",
        "OBELiX composition\nreference",
        "Screen\ncandidate",
    ]
    keyed = disposition.set_index("disposition")
    counts = [int(keyed.loc[item, "n_entries"]) for item in order]
    y_positions = np.arange(4) + 1.35
    disposition_colors = {
        "oxygen_scope_exclusion": FIGURE_1_PALETTE["oxygen_exclusion"],
        "redox_electronic_risk": FIGURE_1_PALETTE["risk_exclusion"],
        "obelix_composition_reference": FIGURE_1_PALETTE["obelix_reference"],
        "screen_candidate": FIGURE_1_PALETTE["candidate"],
    }
    bars = ax.barh(
        y_positions,
        counts,
        color=[disposition_colors[item] for item in order],
        height=0.56,
        zorder=3,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            count + 2,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontweight="bold",
        )
    required = "+".join(str(query_contract.required_elements).split(";"))
    excluded = ", ".join(str(query_contract.excluded_elements).split(";"))
    query_band = FancyBboxPatch(
        (0.014, 0.842),
        0.958,
        0.118,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor=OFF_WHITE,
        edgecolor=LIGHT,
        linewidth=0.7,
        clip_on=False,
        zorder=1,
    )
    ax.add_patch(query_band)
    ax.text(
        0.018,
        0.901,
        f"MP {query_contract.database_version} | {required}; no {excluded} | "
        f"{int(query_contract.min_elements)}–{int(query_contract.max_elements)} elements | "
        rf"{query_contract.min_e_above_hull:g}–"
        rf"{query_contract.max_e_above_hull:g} eV atom$^{{-1}}$ | "
        f"gap≥{query_contract.min_band_gap:g} eV",
        transform=ax.transAxes,
        va="center",
        fontweight="bold",
        fontsize=5.1,
        zorder=2,
    )
    ax.text(
        0.968,
        0.901,
        f"{int(query_contract.n_entries)} entries",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color=WHITE,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.30",
            "fc": FIGURE_1_PALETTE["train_query"],
            "ec": "none",
        },
        zorder=3,
    )
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0, max(counts) * 1.2)
    ax.set_ylim(4.95, -0.35)
    ax.set_xlabel("MP entries (count)")
    _tidy(ax)
    ax_disposition = ax

    ax = fig.add_subplot(bottom[0, 1])
    multiplicity = queue.n_mp_entries.value_counts().sort_index()
    structures = int(np.dot(multiplicity.index.to_numpy(), multiplicity.to_numpy()))
    x_values = multiplicity.index.to_numpy(dtype=int)
    composition_counts = multiplicity.to_numpy(dtype=int)
    _panel(ax, "d", "Structure-to-composition")
    bars = ax.bar(
        x_values,
        composition_counts,
        width=0.58,
        color=FIGURE_1_MULTIPLICITY_COLORS,
        edgecolor=FIGURE_1_PALETTE["candidate"],
        linewidth=0.45,
        zorder=3,
    )
    for bar, multiplicity_value, count in zip(bars, x_values, composition_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 1.2,
            f"{count} comp.\n{count * multiplicity_value} structures",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=5.6,
        )
    formula = " + ".join(
        f"{count}×{multiplicity_value}"
        for multiplicity_value, count in zip(x_values, composition_counts)
    )
    ax.text(
        0.98,
        0.78,
        f"{formula} = {structures}",
        transform=ax.transAxes,
        ha="right",
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.66,
        "Representative\n"
        r"minimum E$_{hull}$; material ID breaks ties",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=GRAY,
    )
    ax.set_xticks(x_values)
    ax.set_xlabel("MP structures per composition (count)")
    ax.set_ylabel("Unique compositions (count)")
    ax.set_ylim(0, 60)
    _tidy(ax)
    _align_columns(fig, (ax_measured, ax_disposition))
    _save(fig, MAIN_FIGURES[0])


def figure_02() -> None:
    targets = _read_csv("fig02a_targets.csv")
    cv = _read_csv("fig02b_grouped_cv.csv")
    test = _read_csv("fig02c_test_predictions.csv")
    baselines = _read_csv("fig02e_strict_baselines.csv")
    sensitivity = _read_csv("fig02d_censor_sensitivity.csv")
    metrics = _metrics()
    fig = _figure(5.55)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.84, 1.16])
    top = outer[0].subgridspec(1, 2, width_ratios=[0.90, 1.10], wspace=0.16)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.18)

    ax = fig.add_subplot(top[0, 0])
    ax_coverage = ax
    _panel(ax, "a", "Target coverage")
    bins = np.linspace(targets.log10_sigma.min(), targets.log10_sigma.max(), 27)
    for split, color, linestyle, label in [
        ("official_train", BLUE, "-", "Train"),
        ("official_test", RED, (0, (4, 2)), "Held-out test"),
    ]:
        values = targets.loc[targets.source_split.eq(split), "log10_sigma"]
        histogram = ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.25,
            ls=linestyle,
            color=color,
            label=f"{label} (n={len(values)})",
        )
        for artist in histogram[2]:
            artist.set_gid(f"target-coverage-{split}")
    censored = targets[targets.censored.astype(bool)]
    ax.scatter(
        censored.log10_sigma,
        np.full(len(censored), 0.004),
        marker="v",
        s=10,
        color=INK,
        clip_on=False,
        label=f"Upper bound (n={len(censored)})",
    )
    ax.set_xlabel(LOG_SIGMA_LABEL)
    ax.set_ylabel("Probability density")
    ax.legend(loc="upper left")
    _tidy(ax)

    ax = fig.add_subplot(top[0, 1])
    ax_cv = ax
    _panel(ax, "b", "Grouped cross-validation")
    model_order = ["Median", "Ridge", "RandomForest", "Perturbation ensemble"]
    model_colors = [MID_GRAY, GRAY, "#555555", BLUE]
    rng = np.random.default_rng(42)
    for x_value, (model, color) in enumerate(zip(model_order, model_colors)):
        values = cv.loc[cv.model.eq(model), "MAE"].to_numpy()
        ax.scatter(
            x_value + rng.uniform(-0.075, 0.075, len(values)),
            values,
            s=15,
            facecolor=WHITE,
            edgecolor=color,
            lw=0.8,
            zorder=3,
        )
        ax.errorbar(
            x_value,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="o",
            ms=4,
            color=color,
            capsize=2.5,
            lw=1.0,
            zorder=4,
        )
    ax.set_xticks(
        range(4),
        ["Median", "Ridge", "Random\nforest", "15-member\nensemble"],
    )
    ax.set_ylabel(MAE_LABEL)
    ax.text(0.98, 0.96, "Open points: five grouped folds", transform=ax.transAxes,
            ha="right", va="top", color=GRAY)
    _tidy(ax)

    c_grid = bottom[0].subgridspec(2, 1, height_ratios=[0.15, 0.85], hspace=0.02)
    header = fig.add_subplot(c_grid[0])
    _panel(header, "c", "Strict held-out performance")
    header.axis("off")
    ci = metrics["cluster_bootstrap"]
    header.text(
        0.0,
        0.10,
        f"n={len(test)}  |  MAE {metrics['strict_test']['MAE']:.2f} "
        f"(95% CI {ci['MAE_ci95'][0]:.2f}–{ci['MAE_ci95'][1]:.2f})\n"
        f"Spearman ρ {metrics['strict_test']['Spearman']:.2f} "
        f"(95% CI {ci['Spearman_ci95'][0]:.2f}–{ci['Spearman_ci95'][1]:.2f})",
        va="bottom",
        color=INK,
    )
    ax = fig.add_subplot(c_grid[1])
    ax_strict = ax
    is_censored = test.censored.astype(bool)
    is_target = test.target_domain.astype(bool)
    other = ~is_censored & ~is_target
    target_exact = ~is_censored & is_target
    ax.scatter(
        test.loc[other, "true_log10_sigma"],
        test.loc[other, "prediction"],
        s=14,
        marker="o",
        facecolor="none",
        edgecolor=MID_GRAY,
        alpha=0.72,
        lw=0.55,
        label=f"Other strict test (n={int(other.sum())})",
    )
    ax.scatter(
        test.loc[target_exact, "true_log10_sigma"],
        test.loc[target_exact, "prediction"],
        s=18,
        marker="D",
        color=BLUE,
        alpha=0.88,
        edgecolor=WHITE,
        lw=0.3,
        label=f"Target-domain exact (n={int(target_exact.sum())})",
    )
    ax.scatter(
        test.loc[is_censored, "true_log10_sigma"],
        test.loc[is_censored, "prediction"],
        s=25,
        marker="v",
        color=RED,
        edgecolor=WHITE,
        lw=0.3,
        label=f"Upper bound (n={int(is_censored.sum())})",
    )
    low = min(test.true_log10_sigma.min(), test.prediction.min()) - 0.4
    high = max(test.true_log10_sigma.max(), test.prediction.max()) + 0.4
    ax.plot([low, high], [low, high], ls="--", lw=0.8, color=GRAY)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(f"Measured {LOG_SIGMA_LABEL}")
    ax.set_ylabel(f"Predicted {LOG_SIGMA_LABEL}")
    ax.legend(loc="lower right", handletextpad=0.35)
    _tidy(ax)

    ax = fig.add_subplot(bottom[0, 1])
    ax_forest = ax
    _panel(ax, "d", "Models and sensitivity")
    model_rows = (
        baselines.loc[baselines["sample"].eq("all_records")]
        .set_index("model")
        .loc[[
            "Median",
            "Ridge",
            "RandomForest",
            "Single base CatBoost",
            "Perturbation ensemble",
        ]]
    )
    sensitivity_values = sensitivity.set_index("definition")["value"]
    target_value = metrics["target_domain_test"]["MAE"]
    rows = [
        ("Median", model_rows.loc["Median", "MAE"], MID_GRAY),
        ("Ridge", model_rows.loc["Ridge", "MAE"], MID_GRAY),
        ("Random forest", model_rows.loc["RandomForest", "MAE"], MID_GRAY),
        ("Single CatBoost", model_rows.loc["Single base CatBoost", "MAE"], GRAY),
        ("15-member ensemble", model_rows.loc["Perturbation ensemble", "MAE"], BLUE),
        ("Point definition", sensitivity_values.loc["point_MAE"], MID_GRAY),
        ("One-sided definition", sensitivity_values.loc["one_sided_MAE"], MID_GRAY),
        ("Uncensored records", sensitivity_values.loc["uncensored_MAE"], MID_GRAY),
        ("Target domain", target_value, RED),
    ]
    y_positions = np.asarray([9, 8, 7, 6, 5, 3, 2, 1, 0], dtype=float)
    ensemble_ci = metrics["cluster_bootstrap"]["MAE_ci95"]
    target_ci = metrics["target_domain_cluster_bootstrap"]["MAE_ci95"]
    ci_right = {5.0: float(ensemble_ci[1]), 0.0: float(target_ci[1])}
    ax.errorbar(
        model_rows.loc["Perturbation ensemble", "MAE"],
        5,
        xerr=[[
            model_rows.loc["Perturbation ensemble", "MAE"] - ensemble_ci[0]
        ], [
            ensemble_ci[1] - model_rows.loc["Perturbation ensemble", "MAE"]
        ]],
        fmt="none",
        color=BLUE,
        lw=1.0,
        capsize=2.5,
        zorder=2,
    )
    ax.errorbar(
        target_value,
        0,
        xerr=[[target_value - target_ci[0]], [target_ci[1] - target_value]],
        fmt="none",
        color=RED,
        lw=1.0,
        capsize=2.5,
        zorder=2,
    )
    for (label, value, color), y_value in zip(rows, y_positions):
        ax.plot(value, y_value, marker="o", ms=4.2, color=color, zorder=3)
        label_x = ci_right.get(float(y_value), float(value)) + 0.025
        ax.text(
            label_x,
            y_value,
            f"{value:.2f}",
            va="center",
            color=INK,
            zorder=4,
        )
    values = np.asarray([row[1] for row in rows], dtype=float)
    ax.axhline(4.05, color=LIGHT, lw=0.8)
    ax.text(values.min() - 0.16, 9.58, "Matched models", fontweight="bold")
    ax.text(values.min() - 0.16, 3.58, "Sensitivity", fontweight="bold")
    ax.text(0.972, 0.018, "Horizontal bars: cluster-bootstrap 95% CI",
            transform=ax.transAxes, ha="right", va="bottom", color=GRAY)
    ax.set_yticks(y_positions, [row[0] for row in rows])
    ax.set_xlabel(MAE_LABEL)
    ax.set_xlim(values.min() - 0.18, max(values.max(), target_ci[1]) + 0.18)
    ax.set_ylim(-1.05, 10.0)
    ax.tick_params(axis="y", length=0)
    _tidy(ax)
    _align_columns(fig, (ax_coverage, header, ax_strict), (ax_cv, ax_forest))
    _align_rows(fig, (ax_coverage, ax_cv), (ax_strict, ax_forest))
    _save(fig, MAIN_FIGURES[1])


def figure_03() -> None:
    ablation = _read_csv("fig03a_representation_ablation.csv")
    convention = _read_csv("fig03b_cell_convention.csv")
    test = _read_csv("fig02c_test_predictions.csv")
    risk = _read_csv("fig03c_risk_correlations.csv")
    fig = _figure(5.55)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.90, 1.10],
        height_ratios=[0.84, 1.16],
        wspace=0.16,
        hspace=0.16,
    )

    ax = fig.add_subplot(grid[0, 0])
    ax_ablation = ax
    _panel(ax, "a", "Representation ablation")
    order = ["M0_formula", "M1_cell_normalized", "M2_legacy"]
    labels = ["M0 · formula only", "M1 · normalized cell", "M2 · legacy raw cell"]
    colors = [BLUE, MID_GRAY, GRAY]
    rng = np.random.default_rng(42)
    for y_value, (name, color) in enumerate(zip(order, colors)):
        values = ablation.loc[ablation.representation.eq(name), "MAE"].to_numpy()
        ax.scatter(
            values,
            y_value + rng.uniform(-0.07, 0.07, len(values)),
            s=14,
            facecolor=WHITE,
            edgecolor=color,
            lw=0.8,
            zorder=3,
        )
        ax.errorbar(
            values.mean(),
            y_value,
            xerr=values.std(ddof=1),
            fmt="o",
            color=color,
            ms=4,
            capsize=2.5,
            zorder=4,
        )
    ax.set_yticks(range(3), labels)
    ax.set_xlabel(f"Grouped-CV {MAE_LABEL}")
    ax.set_ylim(-0.45, 2.45)
    ax.invert_yaxis()
    ax.text(0.98, 0.05, "Open points: five folds", transform=ax.transAxes,
            ha="right", color=GRAY)
    _tidy(ax)

    ax = fig.add_subplot(grid[0, 1])
    ax_convention = ax
    _panel(ax, "b", "Cell-convention audit")
    highlighted = (
        convention.raw_volume_ratio.gt(1.5)
        | convention.volume_per_atom_ratio.gt(1.5)
    )
    ax.scatter(
        convention.loc[~highlighted, "raw_volume_ratio"],
        convention.loc[~highlighted, "volume_per_atom_ratio"],
        s=17,
        marker="o",
        facecolor="none",
        edgecolor=SKY,
        alpha=0.72,
        lw=0.55,
        label="Within threshold",
    )
    ax.scatter(
        convention.loc[highlighted, "raw_volume_ratio"],
        convention.loc[highlighted, "volume_per_atom_ratio"],
        s=21,
        marker="D",
        color=RED,
        alpha=0.88,
        edgecolor=WHITE,
        lw=0.3,
        label="Above 1.5 threshold",
    )
    ax.axvline(1.5, color=GRAY, ls="--", lw=0.8)
    ax.axhline(1.5, color=GRAY, ls="--", lw=0.8)
    ax.plot([1, 5], [1, 5], color=MID_GRAY, ls=":", lw=0.7)
    ax.set_xlim(0.95, max(4.4, convention.raw_volume_ratio.max() * 1.08))
    ax.set_ylim(0.95, max(1.55, convention.volume_per_atom_ratio.max() * 1.08))
    ax.set_xlabel("Raw-cell volume ratio (max/min)")
    ax.set_ylabel("Per-atom volume ratio (max/min)")
    raw_row = convention.loc[convention.raw_volume_ratio.idxmax()]
    normalized_row = convention.loc[convention.volume_per_atom_ratio.idxmax()]
    ax.text(
        0.96,
        0.30,
        f"Raw >1.5: {int(convention.raw_volume_ratio.gt(1.5).sum())} groups",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color=INK,
    )
    ax.annotate(
        "",
        xy=(raw_row.raw_volume_ratio, raw_row.volume_per_atom_ratio),
        xytext=(0.78, 0.45),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.7},
    )
    # Right-aligned so a wider fallback face grows the label into the empty
    # upper-right corner instead of past the canvas edge.
    ax.text(
        0.98,
        0.95,
        "Normalized >1.5: "
        f"{int(convention.volume_per_atom_ratio.gt(1.5).sum())} group",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
    )
    ax.annotate(
        "",
        xy=(normalized_row.raw_volume_ratio, normalized_row.volume_per_atom_ratio),
        xytext=(0.58, 0.87),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.7},
    )
    _tidy(ax)

    ax = fig.add_subplot(grid[1, 0])
    ax_spread = ax
    _panel(ax, "c", "Spread and domain distance")
    scatter = ax.scatter(
        test.spread,
        test.ad_distance,
        c=test.absolute_error,
        cmap="cividis",
        s=23,
        alpha=0.86,
        edgecolor=WHITE,
        lw=0.25,
    )
    ax.set_xlabel(SPREAD_LABEL)
    ax.set_ylabel("Applicability-domain distance (dimensionless)")
    spread_colorbar = _colorbar(
        fig,
        scatter,
        ax,
        ABSOLUTE_ERROR_LABEL,
    )
    rho = test[["spread", "ad_distance"]].corr(method="spearman").iloc[0, 1]
    # Reserve an empty band under the data so the annotation clears every point
    # even when a wider fallback face is substituted for Arial.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom - 0.12 * (top - bottom), top)
    # AD distance is non-negative, so the reserved band carries no tick labels.
    ax.set_yticks([tick for tick in ax.get_yticks() if 0 <= tick <= top])
    ax.text(0.98, 0.035, f"Spread–AD Spearman ρ = {rho:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", color=GRAY)
    _tidy(ax)

    d_grid = grid[1, 1].subgridspec(2, 1, height_ratios=[0.18, 0.82], hspace=0.02)
    header = fig.add_subplot(d_grid[0])
    _panel(header, "d", "Risk–error association")
    signal_order = ["spread_error", "ad_error", "combined_error"]
    signal_labels = ["Ensemble spread", "AD distance", "Combined percentile"]
    base_y = np.arange(3)[::-1].astype(float)
    series = [
        ("oof", -0.18, BLUE, WHITE, "o", "Global OOF"),
        ("test", -0.06, BLUE, BLUE, "o", "Strict test"),
        ("oof_target_domain", 0.06, RED, WHITE, "D", "Target OOF"),
        ("test_target_domain", 0.18, RED, RED, "D", "Target test"),
    ]
    handles = [
        Line2D([0], [0], marker=marker, ls="none", ms=4, mfc=face, mec=color,
               label=label)
        for _, _, color, face, marker, label in series
    ]
    header.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.08),
        ncol=4,
        columnspacing=0.75,
        handletextpad=0.25,
    )
    header.axis("off")
    ax = fig.add_subplot(d_grid[1])
    ax_risk = ax
    for sample, offset, color, facecolor, marker, label in series:
        values = (
            risk.loc[risk["sample"].eq(sample)]
            .set_index("signal")
            .loc[signal_order]
        )
        y_values = base_y + offset
        x_values = values.spearman_rho.to_numpy()
        xerr = np.vstack([
            x_values - values.ci95_low.to_numpy(),
            values.ci95_high.to_numpy() - x_values,
        ])
        errorbar = ax.errorbar(
            x_values,
            y_values,
            xerr=xerr,
            fmt=marker,
            ms=4.0,
            mfc=facecolor,
            mec=color,
            mew=0.85,
            ecolor=color,
            elinewidth=0.8,
            capsize=2.0,
            zorder=3,
            label=label,
        )
        errorbar.lines[0].set_gid(f"risk-series-{sample}")
    # Stop the zero reference short of the bottom strip that carries the
    # resample note, so a wider fallback face cannot run the note into the line.
    ax.axvline(0, ymin=0.10, color=GRAY, ls="--", lw=0.7)
    lower = float(risk.loc[risk.signal.isin(signal_order), "ci95_low"].min())
    upper = float(risk.loc[risk.signal.isin(signal_order), "ci95_high"].max())
    ax.set_xlim(min(-0.25, lower - 0.05), max(0.75, upper + 0.05))
    ax.set_yticks(base_y, signal_labels)
    ax.set_xlabel("Spearman ρ with absolute error (95% CI)")
    # The lower bound leaves a clear strip for the resample note under any face.
    ax.set_ylim(-0.52, 2.48)
    ax.text(0.972, 0.018, "4,000 composition-cluster resamples",
            transform=ax.transAxes, ha="right", va="bottom", color=GRAY)
    _tidy(ax)
    _align_columns(fig, (ax_ablation, ax_spread), (ax_convention, header, ax_risk))
    _align_rows(fig, (ax_ablation, ax_convention), (ax_spread, ax_risk))
    _match_colorbar(spread_colorbar, ax_spread)
    _save(fig, MAIN_FIGURES[2])


def figure_04() -> None:
    projection = _read_csv("fig04a_descriptor_projection.csv")
    ledger = _read_csv("fig04_screen_ledger.csv")
    fig = _figure(5.80)
    outer = fig.add_gridspec(3, 1, height_ratios=[0.11, 0.89, 1.12], hspace=0.08)
    legend_axis = fig.add_subplot(outer[0])
    top = outer[1].subgridspec(1, 3, width_ratios=[1.05, 1.05, 0.90], wspace=0.18)

    ref = projection[projection.source.eq("OBELiX")]
    mp = projection[projection.source.eq("Materials Project")]
    order = [
        "oxygen_scope_exclusion",
        "redox_electronic_risk",
        "obelix_composition_reference",
        "screen_candidate",
    ]
    labels = ["O-scope exclusion", "Redox/electronic risk", "OBELiX reference", "Candidate"]
    handles = [
        Line2D([0], [0], marker="o", ls="none", ms=4,
               color=DISPOSITION_COLORS["reference_measurement"],
               label=f"OBELiX measurements ({len(ref)})")
    ]
    for state, label in zip(order, labels):
        handles.append(Line2D(
            [0],
            [0],
            marker=DISPOSITION_MARKERS[state],
            ls="none",
            ms=4,
            color=DISPOSITION_COLORS[state],
            label=f"{label} ({int(ledger.disposition.eq(state).sum())})",
        ))
    legend_axis.legend(
        handles=handles,
        loc="center",
        ncol=5,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    legend_axis.axis("off")

    ax = fig.add_subplot(top[0, 0])
    ax_descriptor = ax
    _panel(ax, "a", "Descriptor landscape")
    ax.scatter(ref.pc1, ref.pc2, s=4.5,
               color=DISPOSITION_COLORS["reference_measurement"], alpha=0.26,
               rasterized=True)
    plot_order = [
        "oxygen_scope_exclusion",
        "obelix_composition_reference",
        "screen_candidate",
        "redox_electronic_risk",
    ]
    state_alpha = {
        "oxygen_scope_exclusion": 0.56,
        "obelix_composition_reference": 0.72,
        "screen_candidate": 0.68,
        "redox_electronic_risk": 0.94,
    }
    state_zorder = {
        "oxygen_scope_exclusion": 2,
        "obelix_composition_reference": 3,
        "screen_candidate": 4,
        "redox_electronic_risk": 5,
    }
    for state in plot_order:
        part = mp[mp.disposition.eq(state)]
        ax.scatter(
            part.pc1,
            part.pc2,
            s=13,
            marker=DISPOSITION_MARKERS[state],
            color=DISPOSITION_COLORS[state],
            alpha=state_alpha[state],
            edgecolor=WHITE,
            lw=0.2,
            zorder=state_zorder[state],
        )
    ax.set_xlabel("Descriptor PC1 score (dimensionless)")
    ax.set_ylabel("Descriptor PC2 score (dimensionless)")
    _tidy(ax)

    ax = fig.add_subplot(top[0, 1])
    _panel(ax, "b", "Score–domain landscape")
    maximum_spread = ledger.ensemble_spread.max()
    bubble_alpha = {**state_alpha, "screen_candidate": 0.60}
    for state in plot_order:
        part = ledger[ledger.disposition.eq(state)]
        ax.scatter(
            part.ad_distance,
            part.ranking_score,
            s=BUBBLE_MIN_AREA + BUBBLE_SCALE_AREA * part.ensemble_spread / maximum_spread,
            marker=DISPOSITION_MARKERS[state],
            color=DISPOSITION_COLORS[state],
            alpha=bubble_alpha[state],
            edgecolor=WHITE,
            lw=0.2,
            zorder=state_zorder[state],
            label=f"_bubble_{state}",
        )
    spread_quantiles = ledger.ensemble_spread.quantile([0.25, 0.50, 0.75])
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="none",
            mfc=WHITE,
            mec=GRAY,
            mew=0.7,
            ms=np.sqrt(BUBBLE_MIN_AREA + BUBBLE_SCALE_AREA * value / maximum_spread),
            label=f"q{int(quantile * 100)} {value:.2f}",
        )
        for quantile, value in spread_quantiles.items()
    ]
    ax.legend(handles=size_handles, title="Ensemble spread\n(log$_{10}$ units)", loc="lower left",
              handletextpad=0.4, borderaxespad=0.2)
    ax.set_xscale("log")
    positive_distances = ledger.ad_distance[ledger.ad_distance > 0]
    ax.set_xlim(
        positive_distances.min() * 0.72,
        positive_distances.max() * 1.25,
    )
    ax.set_xlabel("Applicability-domain distance (dimensionless; log scale)")
    ax.set_ylabel(RANKING_SCORE_LABEL)
    _tidy(ax)

    ax = fig.add_subplot(top[0, 2])
    _panel(ax, "c", "AD by disposition")
    transformed = [
        np.log10(1 + ledger.loc[ledger.disposition.eq(state), "ad_distance"])
        for state in order
    ]
    box = ax.boxplot(
        transformed,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": INK, "lw": 0.9},
        whiskerprops={"color": GRAY, "lw": 0.7},
        capprops={"color": GRAY, "lw": 0.7},
    )
    rng = np.random.default_rng(42)
    for x_value, (patch, state, values) in enumerate(zip(box["boxes"], order, transformed), 1):
        patch_alpha = 0.36 if state == "obelix_composition_reference" else 0.28
        point_alpha = 0.40 if state == "obelix_composition_reference" else 0.28
        patch.set_facecolor(mpl.colors.to_rgba(DISPOSITION_COLORS[state], patch_alpha))
        patch.set_edgecolor(DISPOSITION_COLORS[state])
        patch.set_linewidth(0.7)
        ax.scatter(
            x_value + rng.uniform(-0.13, 0.13, len(values)),
            values,
            s=4.0,
            color=DISPOSITION_COLORS[state],
            alpha=point_alpha,
            edgecolor="none",
            zorder=2,
        )
    ax.set_xticks(range(1, 5), ["O-scope", "Redox", "Reference", "Candidate"],
                  rotation=25, ha="right")
    ax.set_ylabel(r"log$_{10}$(1 + AD distance)")
    _tidy(ax)

    ax = fig.add_subplot(outer[2])
    ax_trace = ax
    _panel(ax, "d", "Complete 248-entry rank trace")
    ranked = ledger.sort_values("screen_rank")
    ax.plot(ranked.screen_rank, ranked.ranking_score, color=LIGHT, lw=0.75, zorder=0)
    for state in plot_order:
        part = ranked[ranked.disposition.eq(state)]
        ax.scatter(
            part.screen_rank,
            part.ranking_score,
            s=13,
            marker=DISPOSITION_MARKERS[state],
            color=DISPOSITION_COLORS[state],
            alpha=state_alpha[state],
            edgecolor=WHITE,
            lw=0.2,
            zorder=state_zorder[state],
        )
    ax.set_xlabel("Global MP-entry rank")
    ax.set_ylabel(RANKING_SCORE_LABEL)
    ax.set_xlim(-2, 254)
    _tidy(ax)
    _align_columns(fig, (ax_descriptor, ax_trace))
    _save(fig, MAIN_FIGURES[3])


def figure_05() -> None:
    queue = _read_csv("fig05_candidate_queue.csv").sort_values("candidate_rank")
    fig = _figure(5.80)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.84, 1.16],
        height_ratios=[0.90, 1.10],
        wspace=0.20,
        hspace=0.16,
    )

    ax = fig.add_subplot(grid[0, 0])
    ax_atlas = ax
    _panel(ax, "a", "Candidate atlas", title_x=0.04)
    features = [
        "ranking_score",
        "ensemble_spread",
        "ad_distance",
        "min_e_above_hull",
        "max_band_gap",
    ]
    labels = ["Score", "Spread", "AD", r"E$_{hull}$", "Band gap"]
    matrix = queue[features].copy()
    center = matrix.median()
    scale = matrix.quantile(0.75) - matrix.quantile(0.25)
    scale = scale.where(scale > 0, matrix.std()).replace(0, 1)
    matrix = (matrix - center) / scale
    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=DIVERGING_CMAP,
        norm=TwoSlopeNorm(vcenter=0, vmin=-HEATMAP_LIMIT, vmax=HEATMAP_LIMIT),
    )
    overrange = matrix.to_numpy() > HEATMAP_LIMIT
    for row, column in np.argwhere(overrange):
        outline = Rectangle(
            (column - 0.5, row - 0.5),
            1,
            1,
            fill=False,
            edgecolor=INK,
            linewidth=0.35,
            zorder=3,
        )
        outline.set_gid("heatmap-overrange")
        ax.add_patch(outline)
    ax.set_xticks(np.arange(5), labels, rotation=28, ha="right")
    ax.set_yticks([0, 14, 29, 44, 61], ["1", "15", "30", "45", "62"])
    ax.set_ylabel("Candidate rank")
    _colorbar(
        fig,
        image,
        ax,
        "Column-wise deviation from median (IQR units)",
        extend="max",
    )
    ax = fig.add_subplot(grid[0, 1])
    ax_consensus = ax
    _panel(ax, "b", "Consensus ranking")
    state_order = ["extrapolative", "high_score_lower_risk", "unflagged"]
    ax.plot(queue.candidate_rank, queue.ranking_score, color=LIGHT, lw=0.75, zorder=0)
    plot_state_order = ["unflagged", "high_score_lower_risk", "extrapolative"]
    state_alpha = {
        "unflagged": 0.68,
        "high_score_lower_risk": 0.82,
        "extrapolative": 0.94,
    }
    for zorder, state in enumerate(plot_state_order, 2):
        part = queue[queue.evidence_state.eq(state)]
        ax.scatter(
            part.candidate_rank,
            part.ranking_score,
            s=22,
            marker=STATE_MARKERS[state],
            color=STATE_COLORS[state],
            alpha=state_alpha[state],
            zorder=zorder,
            edgecolor=WHITE,
            lw=0.3,
            label=f"{state.replace('_', ' ')} ({len(part)})",
        )
    ax.set_xlabel("Candidate composition rank")
    ax.set_ylabel(RANKING_SCORE_LABEL)
    ax.set_xlim(0, 63)
    legend_labels = [
        f"{state.replace('_', ' ')} ({int(queue.evidence_state.eq(state).sum())})"
        for state in state_order
    ]
    legend_handles, plotted_labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(plotted_labels, legend_handles))
    ax.legend(
        [handle_by_label[label] for label in legend_labels],
        legend_labels,
        loc="upper right",
        handletextpad=0.35,
    )
    _tidy(ax)

    ax = fig.add_subplot(grid[1, 0])
    ax_member_map = ax
    _panel(ax, "c", "Spread–domain map", title_x=0.04)
    norm = Normalize(vmin=queue.ranking_score.min(), vmax=queue.ranking_score.max())
    scatter = None
    for zorder, state in enumerate(plot_state_order, 2):
        part = queue[queue.evidence_state.eq(state)]
        scatter = ax.scatter(
            part.ensemble_spread,
            part.ad_distance,
            c=part.ranking_score,
            cmap="viridis",
            norm=norm,
            marker=STATE_MARKERS[state],
            s=28,
            edgecolor=INK,
            lw=0.35,
            alpha=state_alpha[state],
            zorder=zorder,
        )
    ax.set_xlabel(SPREAD_LABEL)
    ax.set_ylabel("Applicability-domain distance (dimensionless)")
    if scatter is None:
        raise RuntimeError("candidate queue contains no evidence states")
    _colorbar(fig, scatter, ax, RANKING_SCORE_LABEL)
    _tidy(ax)

    ax = fig.add_subplot(grid[1, 1])
    ax_stability = ax
    _panel(ax, "d", "Member-rank stability")
    x = queue.candidate_rank.to_numpy()
    q25 = queue.member_rank_q25.to_numpy()
    median = queue.member_rank_median.to_numpy()
    q75 = queue.member_rank_q75.to_numpy()
    ax.vlines(x, q25, q75, color=MID_GRAY, alpha=0.62, lw=0.65, zorder=1)
    scatter = ax.scatter(
        x,
        median,
        c=queue.top_quartile_frequency,
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=19,
        edgecolor=WHITE,
        lw=0.3,
        zorder=2,
    )
    ax.plot([1, len(queue)], [1, len(queue)], ls="--", lw=0.75, color=INK, zorder=0)
    handles = [
        Line2D([0], [0], color=MID_GRAY, lw=0.8, label="Member rank q25–q75"),
        Line2D([0], [0], color=INK, lw=0.8, ls="--", label="Consensus identity"),
    ]
    ax.legend(handles=handles, loc="upper right", handlelength=1.6)
    ax.set_xlabel("Final consensus candidate rank")
    ax.set_ylabel("Median rank across 15 members")
    ax.set_xlim(0, len(queue) + 1)
    ax.set_ylim(len(queue) + 1, 0)
    _colorbar(fig, scatter, ax, "Top-quartile frequency (fraction)")
    _tidy(ax)
    _align_columns(fig, (ax_atlas, ax_member_map), (ax_consensus, ax_stability))
    _save(fig, MAIN_FIGURES[4])


def figure_S1() -> None:
    """Supplementary Figure S1: boosting curves (a) and a learning curve (b)."""
    curves = _read_csv("figS1a_boosting_curves.csv")
    learning = _read_csv("figS1b_learning_curve.csv")
    metrics = _metrics()
    fig = _figure(2.75)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.18)

    ax = fig.add_subplot(grid[0, 0])
    _panel(ax, "a", "Boosting curves")
    config_colors = {"compact": TEAL, "base": BLUE, "deep_slow": PURPLE}
    specs = {spec.name: spec for spec in MODEL_SPECS}
    train_style = (0, (3, 1.5))
    handles = []
    for name, color in config_colors.items():
        part = curves.loc[curves.config.eq(name)]
        summary = part.groupby("iteration").agg(
            train_mean=("train_rmse", "mean"),
            validation_mean=("validation_rmse", "mean"),
            validation_sd=("validation_rmse", lambda values: values.std(ddof=1)),
        )
        x = summary.index.to_numpy()
        ax.fill_between(
            x,
            summary.validation_mean - summary.validation_sd,
            summary.validation_mean + summary.validation_sd,
            color=color,
            alpha=0.14,
            lw=0,
        )
        ax.plot(x, summary.validation_mean, color=color, lw=1.1,
                label=f"_validation_{name}", gid=f"boosting-validation-{name}")
        ax.plot(x, summary.train_mean, color=color, lw=0.8, ls=train_style,
                label=f"_train_{name}", gid=f"boosting-train-{name}")
        spec = specs[name]
        handles.append(Line2D(
            [0], [0], color=color, lw=1.1,
            label=(f"{name.replace('_', '-')}: {spec.iterations} iterations, "
                   f"depth {spec.depth}, rate {spec.learning_rate:g}"),
        ))
    handles.append(Line2D([0], [0], color=INK, lw=1.1, label="Validation fold, mean ± SD"))
    handles.append(Line2D([0], [0], color=INK, lw=0.8, ls=train_style, label="Training fold, mean"))
    ax.legend(handles=handles, loc="upper right", handlelength=2.2, borderaxespad=0.3)
    ax.set_xlim(0, curves.iteration.max() * 1.03)
    ax.set_ylim(0, curves[["train_rmse", "validation_rmse"]].to_numpy().max() * 1.55)
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel(r"RMSE (log$_{10}$ units)")
    _tidy(ax)

    ax = fig.add_subplot(grid[0, 1])
    _panel(ax, "b", "Learning curve")
    summary = learning.groupby("fraction").agg(
        n_records=("n_train_records", "mean"),
        mae_mean=("MAE", "mean"),
        mae_sd=("MAE", lambda values: values.std(ddof=1) if len(values) > 1 else 0.0),
    )
    ax.scatter(
        learning.n_train_records,
        learning.MAE,
        s=12,
        facecolor=WHITE,
        edgecolor=BLUE,
        lw=0.7,
        zorder=3,
        label="Individual composition-grouped subsamples",
    )
    ax.errorbar(
        summary.n_records,
        summary.mae_mean,
        yerr=summary.mae_sd,
        fmt="o-",
        color=BLUE,
        ms=3.5,
        lw=1.0,
        capsize=2,
        zorder=4,
        label="Single base CatBoost, mean ± SD",
    )
    ensemble_mae = metrics["strict_test"]["MAE"]
    single_mae = metrics["strict_test_baselines"]["Single base CatBoost"]["all_records"]["MAE"]
    ax.axhline(ensemble_mae, color=GRAY, ls="--", lw=0.8, gid="reference-ensemble",
               label=f"15-member ensemble, all 476 records ({ensemble_mae:.2f})")
    ax.axhline(single_mae, color=MID_GRAY, ls=":", lw=0.9, gid="reference-single",
               label=f"Single base CatBoost, all 476 records ({single_mae:.2f})")
    ax.legend(loc="upper right", borderaxespad=0.3)
    ax.set_xlim(0, learning.n_train_records.max() * 1.06)
    ax.set_ylim(min(learning.MAE.min(), single_mae) * 0.85, learning.MAE.max() * 1.42)
    ax.set_xlabel("Strict-training records used (n)")
    ax.set_ylabel(f"Strict-test {MAE_LABEL}")
    _tidy(ax)

    _save(fig, SUPPLEMENTARY_FIGURES[0], directory=SUPPLEMENTARY)


def render_figures(figure: int | None = None) -> None:
    """Render the five main figures plus Supplementary Figure S1, or one main figure."""
    _style()
    functions = [figure_01, figure_02, figure_03, figure_04, figure_05]
    if figure is None:
        _clean_figure_directory()
        _clean_supplementary_directory()
        for function in functions:
            function()
        figure_S1()
        return
    if figure < 1 or figure > len(functions):
        raise ValueError("figure must be between 1 and 5")
    functions[figure - 1]()


def render_supplementary() -> None:
    """Render only the supplementary figure(s) into figures/supplementary/."""
    _style()
    _clean_supplementary_directory()
    figure_S1()
