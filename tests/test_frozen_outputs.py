import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.path import Path as MatplotlibPath
from matplotlib.transforms import Bbox
from PIL import Image

from p1screen import figures as figure_module
from p1screen.config import (
    CANDIDATE_QUEUE,
    FIGURES,
    LEARNING_CURVE_FRACTIONS,
    LEARNING_CURVE_REPEATS,
    MAIN_FIGURES,
    METRICS,
    MODEL_MANIFEST,
    MODEL_SPECS,
    MP_LEDGER,
    MP_QUERY,
    MP_SNAPSHOT,
    N_FOLDS,
    RELEASE_MANIFEST,
    SOURCE_DATA,
    SUPPLEMENTARY,
    SUPPLEMENTARY_FIGURES,
)
from p1screen.verify import verify_release


def _capture_figures(monkeypatch):
    captured = {}

    def capture(fig, name, directory=None):
        captured[name] = fig

    monkeypatch.setattr(figure_module, "_save", capture)
    figure_module._style()
    for function in (
        figure_module.figure_01,
        figure_module.figure_02,
        figure_module.figure_03,
        figure_module.figure_04,
        figure_module.figure_05,
        figure_module.figure_S1,
    ):
        function()
    for fig in captured.values():
        fig.canvas.draw()
    return captured


def _axis_with_title(fig, title):
    return next(
        ax
        for ax in fig.axes
        if any(ax.get_title(loc=loc) == title for loc in ("left", "center", "right"))
    )


def _collection_with_label(ax, prefix):
    return next(
        collection
        for collection in ax.collections
        if collection.get_label().startswith(prefix)
    )


def _path_intersects_bbox(points, bbox):
    return len(points) >= 2 and MatplotlibPath(points).intersects_bbox(
        bbox,
        filled=False,
    )


def _legend_data_collisions(ax, renderer):
    legend = ax.get_legend()
    if legend is None:
        return 0
    bbox = legend.get_window_extent(renderer)
    collisions = 0
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            offsets = np.asarray(collection.get_offsets())
            if offsets.size == 0:
                continue
            points = collection.get_offset_transform().transform(offsets)
            sizes = np.asarray(collection.get_sizes())
            if sizes.size == 0:
                sizes = np.zeros(len(points))
            elif sizes.size == 1:
                sizes = np.repeat(sizes, len(points))
            for point, size in zip(points, sizes):
                radius = np.sqrt(float(size)) * ax.figure.dpi / 144
                padded = Bbox.from_extents(
                    bbox.x0 - radius,
                    bbox.y0 - radius,
                    bbox.x1 + radius,
                    bbox.y1 + radius,
                )
                collisions += int(padded.contains(float(point[0]), float(point[1])))
        elif isinstance(collection, LineCollection):
            collisions += sum(
                _path_intersects_bbox(
                    collection.get_transform().transform(segment),
                    bbox,
                )
                for segment in collection.get_segments()
            )
    for line in ax.lines:
        points = line.get_transform().transform(
            np.column_stack([line.get_xdata(orig=False), line.get_ydata(orig=False)])
        )
        collisions += sum(
            _path_intersects_bbox(points[index:index + 2], bbox)
            for index in range(max(0, len(points) - 1))
        )
    return collisions


def _free_text_data_collisions(ax, renderer):
    point_centres = []
    line_segments = []
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            offsets = np.asarray(collection.get_offsets())
            if offsets.size:
                point_centres.extend(
                    collection.get_offset_transform().transform(offsets)
                )
        elif isinstance(collection, LineCollection):
            line_segments.extend(
                collection.get_transform().transform(segment)
                for segment in collection.get_segments()
            )
    for line in ax.lines:
        points = line.get_transform().transform(
            np.column_stack([line.get_xdata(orig=False), line.get_ydata(orig=False)])
        )
        line_segments.extend(
            points[index:index + 2]
            for index in range(max(0, len(points) - 1))
        )
    collisions = 0
    for text in ax.texts:
        label = text.get_text().strip()
        if not label or label in {"a", "b", "c", "d", "e"}:
            continue
        bbox = text.get_window_extent(renderer)
        collisions += sum(
            bbox.contains(float(point[0]), float(point[1]))
            for point in point_centres
        )
        collisions += sum(
            _path_intersects_bbox(segment, bbox)
            for segment in line_segments
        )
    return collisions


def test_frozen_screen_accounting():
    snapshot = pd.read_csv(MP_SNAPSHOT)
    ledger = pd.read_csv(MP_LEDGER)
    queue = pd.read_csv(CANDIDATE_QUEUE)
    assert len(snapshot) == len(ledger) == 248
    assert snapshot.material_id.nunique() == 248
    assert ledger.disposition.value_counts().to_dict() == {
        "oxygen_scope_exclusion": 80,
        "screen_candidate": 79,
        "obelix_composition_reference": 70,
        "redox_electronic_risk": 19,
    }
    assert len(queue) == queue.composition_key.nunique() == 62
    assert queue.n_mp_entries.value_counts().sort_index().to_dict() == {1: 49, 2: 9, 3: 4}
    assert queue.n_mp_entries.sum() == 79
    assert {
        "member_rank_q25", "member_rank_median", "member_rank_q75",
        "member_rank_iqr", "top_quartile_frequency",
    } <= set(queue.columns)
    assert (queue.member_rank_q25 <= queue.member_rank_median).all()
    assert (queue.member_rank_median <= queue.member_rank_q75).all()
    assert np.allclose(
        queue.member_rank_q75 - queue.member_rank_q25,
        queue.member_rank_iqr,
        rtol=0,
        atol=1e-12,
    )
    assert queue.evidence_state.value_counts().to_dict() == {
        "extrapolative": 27,
        "unflagged": 23,
        "high_score_lower_risk": 12,
    }


def test_metrics_and_model_manifest_are_strict_json():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    assert metrics["strict_test"]["MAE"] > 0
    assert metrics["split_audit"]["n_train_strict"] == 476
    assert metrics["target_domain_test"]["n_records"] == 24
    assert "strict_test_baselines" in metrics
    assert "strict_model_comparisons" in metrics
    assert "training_sensitivity" in metrics
    assert "risk_correlation_bootstrap" in metrics
    assert metrics["target_domain_cluster_bootstrap"]["n_groups"] == 23
    assert manifest["n_members"] == len(manifest["members"]) == 15
    assert manifest["n_features"] == 134
    sensitivity = pd.DataFrame(manifest["route_sensitivity"])
    assert set(sensitivity["quantile"].unique()) == {0.70, 0.75, 0.80}
    assert sensitivity.groupby("quantile")["count"].sum().eq(62).all()
    assert (
        sensitivity.loc[sensitivity["quantile"].eq(0.75)]
        .set_index("state")["count"]
        .to_dict()
    ) == {
        "extrapolative": 27,
        "high_score_lower_risk": 12,
        "unflagged": 23,
    }


def test_figure_1_query_contract_is_generated_from_frozen_query():
    query = json.loads(MP_QUERY.read_text(encoding="utf-8"))
    contract = pd.read_csv(SOURCE_DATA / "fig01c_mp_query_contract.csv").iloc[0]
    filters = query["filters"]
    assert contract.database_version == query["database_version"]
    assert contract.n_entries == query["n_entries"] == 248
    assert contract.required_elements == ";".join(filters["elements"])
    assert contract.excluded_elements == ";".join(filters["exclude_elements"])
    assert [contract.min_elements, contract.max_elements] == filters["num_elements"]
    assert [contract.min_e_above_hull, contract.max_e_above_hull] == (
        filters["energy_above_hull_eV_atom"]
    )
    assert contract.min_band_gap == filters["band_gap_eV"][0]


def test_risk_correlation_bootstrap_source_contract():
    risk = pd.read_csv(SOURCE_DATA / "fig03c_risk_correlations.csv")
    assert len(risk) == 16
    assert {
        "sample", "signal", "spearman_rho", "ci95_low", "ci95_high",
        "n_records", "n_compositions", "n_bootstrap", "rank_normalization",
    } <= set(risk.columns)
    assert np.isfinite(risk[["spearman_rho", "ci95_low", "ci95_high"]]).all().all()
    assert (risk.ci95_low <= risk.ci95_high).all()
    expected = {
        "oof": (478, 390, "within_fold"),
        "test": (121, 112, "within_sample"),
        "oof_target_domain": (117, 97, "within_fold"),
        "test_target_domain": (24, 23, "within_sample"),
    }
    for sample, (records, compositions, normalization) in expected.items():
        rows = risk.loc[risk["sample"].eq(sample)]
        assert set(rows.n_records) == {records}
        assert set(rows.n_compositions) == {compositions}
        assert set(rows.n_bootstrap) == {4000}
        assert set(rows.rank_normalization) == {normalization}


def test_nature_style_contract():
    figure_module._style()
    assert mpl.rcParams["font.sans-serif"][0] == "Arial"
    for key in (
        "font.size", "axes.labelsize", "axes.titlesize",
        "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
    ):
        assert 5 <= float(mpl.rcParams[key]) <= 7
    assert figure_module.PANEL_LABEL_SIZE == 8
    assert not mpl.rcParams["axes.grid"]
    assert not mpl.rcParams["legend.frameon"]


def test_figure_objects_have_no_avoidable_collisions_or_clipping(monkeypatch):
    captured = _capture_figures(monkeypatch)
    try:
        for name, fig in captured.items():
            renderer = fig.canvas.get_renderer()
            assert sum(
                _legend_data_collisions(ax, renderer) for ax in fig.axes
            ) == 0, f"legend overlaps plotted evidence in {name}"
            assert sum(
                _free_text_data_collisions(ax, renderer) for ax in fig.axes
            ) == 0, f"free text overlaps plotted evidence in {name}"
            width, height = renderer.width, renderer.height
            for text in fig.findobj(mpl.text.Text):
                if (
                    not text.get_visible()
                    or not text.get_text().strip()
                    or text.get_clip_on()
                ):
                    continue
                bbox = text.get_window_extent(renderer)
                assert bbox.x0 >= -2 and bbox.x1 <= width + 2, (
                    f"text is horizontally clipped in {name}: {text.get_text()}"
                )
                assert bbox.y0 >= -2 and bbox.y1 <= height + 2, (
                    f"text is vertically clipped in {name}: {text.get_text()}"
                )

        figure_1 = captured[MAIN_FIGURES[0]]
        measured = _axis_with_title(figure_1, "Measured evidence")
        assert [patch.get_facecolor() for patch in measured.patches[:2]] == [
            mpl.colors.to_rgba(figure_module.FIGURE_1_PALETTE["train_query"]),
            mpl.colors.to_rgba(figure_module.FIGURE_1_PALETTE["test_risk"]),
        ]

        split = _axis_with_title(figure_1, "Leakage-corrected split")
        expected_box_colors = {
            "Official train": figure_module.FIGURE_1_PALETTE["train_query"],
            "Official test": figure_module.FIGURE_1_PALETTE["test_risk"],
            "2 shared compositions": figure_module.FIGURE_1_PALETTE["test_risk"],
            "Strict protocol": figure_module.FIGURE_1_PALETTE["train_query"],
        }
        for prefix, color in expected_box_colors.items():
            text = next(item for item in split.texts if item.get_text().startswith(prefix))
            assert text.get_bbox_patch().get_edgecolor() == mpl.colors.to_rgba(color)

        disposition = _axis_with_title(figure_1, "Frozen MP query and disposition")
        disposition_keys = (
            "oxygen_exclusion",
            "risk_exclusion",
            "obelix_reference",
            "candidate",
        )
        assert [patch.get_facecolor() for patch in disposition.patches[:4]] == [
            mpl.colors.to_rgba(figure_module.FIGURE_1_PALETTE[key])
            for key in disposition_keys
        ]
        badge = next(item for item in disposition.texts if item.get_text() == "248 entries")
        assert badge.get_bbox_patch().get_facecolor() == mpl.colors.to_rgba(
            figure_module.FIGURE_1_PALETTE["train_query"]
        )

        collapse = _axis_with_title(figure_1, "Structure-to-composition")
        assert [patch.get_facecolor() for patch in collapse.patches[:3]] == [
            mpl.colors.to_rgba(color)
            for color in figure_module.FIGURE_1_MULTIPLICITY_COLORS
        ]
        assert all(
            patch.get_edgecolor()
            == mpl.colors.to_rgba(figure_module.FIGURE_1_PALETTE["candidate"])
            for patch in collapse.patches[:3]
        )

        figure_2 = captured[MAIN_FIGURES[1]]
        coverage = _axis_with_title(figure_2, "Target coverage")
        train_step = next(
            patch
            for patch in coverage.patches
            if patch.get_gid() == "target-coverage-official_train"
        )
        test_step = next(
            patch
            for patch in coverage.patches
            if patch.get_gid() == "target-coverage-official_test"
        )
        assert train_step.get_linestyle() != test_step.get_linestyle()

        strict = next(
            ax for ax in figure_2.axes if ax.get_xlabel().startswith("Measured")
        )
        other = _collection_with_label(strict, "Other strict test")
        target = _collection_with_label(strict, "Target-domain exact")
        assert not np.array_equal(
            other.get_paths()[0].vertices,
            target.get_paths()[0].vertices,
        )
        assert other.get_facecolors().size == 0
        assert target.get_facecolors().size > 0

        figure_3 = captured[MAIN_FIGURES[2]]
        convention = _axis_with_title(figure_3, "Cell-convention audit")
        within = _collection_with_label(convention, "Within threshold")
        above = _collection_with_label(convention, "Above 1.5 threshold")
        assert not np.array_equal(
            within.get_paths()[0].vertices,
            above.get_paths()[0].vertices,
        )
        assert within.get_facecolors().size == 0
        assert above.get_facecolors().size > 0

        risk_axis = next(
            ax
            for ax in figure_3.axes
            if ax.get_xlabel().startswith("Spearman")
        )
        risk_lines = {
            line.get_gid(): line
            for line in risk_axis.lines
            if line.get_gid() is not None and line.get_gid().startswith("risk-series-")
        }
        assert risk_lines["risk-series-oof"].get_marker() == "o"
        assert risk_lines["risk-series-test"].get_marker() == "o"
        assert risk_lines["risk-series-oof_target_domain"].get_marker() == "D"
        assert risk_lines["risk-series-test_target_domain"].get_marker() == "D"
        assert mpl.colors.to_rgba(
            risk_lines["risk-series-oof"].get_markerfacecolor()
        ) == mpl.colors.to_rgba(figure_module.WHITE)
        assert mpl.colors.to_rgba(
            risk_lines["risk-series-test"].get_markerfacecolor()
        ) == mpl.colors.to_rgba(figure_module.BLUE)

        figure_4 = captured[MAIN_FIGURES[3]]
        bubble_axis = _axis_with_title(figure_4, "Score–domain landscape")
        ledger = pd.read_csv(SOURCE_DATA / "fig04_screen_ledger.csv")
        maximum_spread = ledger.ensemble_spread.max()
        for state in figure_module.DISPOSITION_MARKERS:
            collection = _collection_with_label(bubble_axis, f"_bubble_{state}")
            values = ledger.loc[ledger.disposition.eq(state), "ensemble_spread"]
            expected = (
                figure_module.BUBBLE_MIN_AREA
                + figure_module.BUBBLE_SCALE_AREA * values / maximum_spread
            )
            assert np.allclose(collection.get_sizes(), expected)
        assert max(
            float(collection.get_sizes().max())
            for collection in bubble_axis.collections
        ) <= figure_module.BUBBLE_MIN_AREA + figure_module.BUBBLE_SCALE_AREA

        figure_5 = captured[MAIN_FIGURES[4]]
        atlas = _axis_with_title(figure_5, "Candidate atlas")
        assert sum(
            patch.get_gid() == "heatmap-overrange" for patch in atlas.patches
        ) == 8
        assert atlas.images[0].colorbar.extend == "max"

        figure_s1 = captured[SUPPLEMENTARY_FIGURES[0]]
        boosting = _axis_with_title(figure_s1, "Boosting curves")
        validation_lines = [
            line for line in boosting.lines
            if str(line.get_gid()).startswith("boosting-validation-")
        ]
        training_lines = [
            line for line in boosting.lines
            if str(line.get_gid()).startswith("boosting-train-")
        ]
        assert len(validation_lines) == len(MODEL_SPECS)
        assert len(training_lines) == len(MODEL_SPECS)
        assert all(line.get_linestyle() == "-" for line in validation_lines)
        assert all(line.get_linestyle() != "-" for line in training_lines)
        learning_axis = _axis_with_title(figure_s1, "Learning curve")
        reference_styles = {
            line.get_gid(): line.get_linestyle()
            for line in learning_axis.lines
            if str(line.get_gid()).startswith("reference-")
        }
        assert reference_styles == {"reference-ensemble": "--", "reference-single": ":"}
    finally:
        for fig in captured.values():
            plt.close(fig)


def test_scientific_axis_label_contract():
    source = Path(figure_module.__file__).read_text(encoding="utf-8")
    assert "rank (count)" not in source
    assert "standardized units" not in source
    assert 'log$_{10}$ S cm$^{-1}$' not in source
    assert figure_module.LOG_SIGMA_LABEL in source


def _assert_raster_contract(path):
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.width == 6480
        assert image.height <= 6024
        assert all(abs(value - 900) <= 1 for value in image.info["dpi"])
        assert image.getpixel((0, 0)) == (255, 255, 255)
    assert path.stat().st_size <= 50 * 1024 * 1024


def test_exact_figure_inventory_and_raster_contract():
    files = sorted(path.name for path in FIGURES.iterdir() if path.is_file())
    assert files == sorted(MAIN_FIGURES)
    for name in MAIN_FIGURES:
        _assert_raster_contract(FIGURES / name)
    supplementary = sorted(path.name for path in SUPPLEMENTARY.iterdir() if path.is_file())
    assert supplementary == sorted(SUPPLEMENTARY_FIGURES)
    for name in SUPPLEMENTARY_FIGURES:
        _assert_raster_contract(SUPPLEMENTARY / name)


def test_supplementary_source_contract():
    boosting = pd.read_csv(SOURCE_DATA / "figS1a_boosting_curves.csv")
    assert {
        "config", "fold", "iteration", "train_rmse", "validation_rmse",
        "n_train", "n_validation",
    } <= set(boosting.columns)
    assert set(boosting.config) == {spec.name for spec in MODEL_SPECS}
    lengths = boosting.groupby(["config", "fold"]).iteration.agg(["size", "max"])
    for spec in MODEL_SPECS:
        per_fold = lengths.loc[spec.name]
        assert len(per_fold) == N_FOLDS
        assert set(per_fold["size"]) == {spec.iterations}
        assert set(per_fold["max"]) == {spec.iterations}
    assert np.isfinite(boosting[["train_rmse", "validation_rmse"]]).all().all()
    # Boosting must reduce training error monotonically at the fold-mean level.
    for spec in MODEL_SPECS:
        mean_train = boosting.loc[boosting.config.eq(spec.name)].groupby("iteration").train_rmse.mean()
        assert mean_train.iloc[-1] < mean_train.iloc[0]

    learning = pd.read_csv(SOURCE_DATA / "figS1b_learning_curve.csv")
    assert {
        "fraction", "repeat_seed", "n_train_records", "n_train_compositions",
        "MAE", "RMSE", "R2", "Spearman", "MAE_target_domain",
    } <= set(learning.columns)
    assert np.allclose(sorted(learning.fraction.unique()), LEARNING_CURVE_FRACTIONS)
    assert len(learning) == (len(LEARNING_CURVE_FRACTIONS) - 1) * LEARNING_CURVE_REPEATS + 1
    full = learning.loc[np.isclose(learning.fraction, 1.0)]
    assert len(full) == 1
    assert int(full.n_train_records.iloc[0]) == 476
    assert int(full.n_train_compositions.iloc[0]) == 388
    assert np.isfinite(learning[["MAE", "RMSE", "R2", "Spearman", "MAE_target_domain"]]).all().all()
    assert learning.n_train_records.is_monotonic_increasing or (
        learning.groupby("fraction").n_train_records.mean().is_monotonic_increasing
    )


def test_frozen_verification_is_read_only():
    before = RELEASE_MANIFEST.read_bytes()
    verify_release(frozen=True)
    assert RELEASE_MANIFEST.read_bytes() == before
