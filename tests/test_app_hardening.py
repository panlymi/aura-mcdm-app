from __future__ import annotations

import numpy as np
from streamlit.testing.v1 import AppTest

from mcdm.uploads import MAX_UPLOAD_BYTES


CRISP_MATRIX = (
    b"Alternative,C1,C2,C3\n"
    b"A1,9,2,6\n"
    b"A2,7,4,8\n"
    b"A3,5,7,5\n"
)


def _app_with_upload(content: bytes, filename: str = "matrix.csv") -> AppTest:
    app = AppTest.from_file("aura_app.py", default_timeout=60).run()
    app.file_uploader[0].set_value((filename, content, "text/csv")).run()
    return app


def _click_button(app: AppTest, label: str) -> AppTest:
    button = next(widget for widget in app.button if widget.label == label)
    return button.click().run()


def _run_aura_baseline(app: AppTest) -> AppTest:
    button = next(
        widget for widget in app.button if "Run AURA Calculation" in widget.label
    )
    return button.click().run()


def _warning_contains(app: AppTest, text: str) -> bool:
    return any(text in warning.value for warning in app.warning)


def test_malformed_upload_has_a_friendly_error_without_raw_exception():
    app = _app_with_upload(b"\xff\xfe\x00not-a-csv", "broken.CSV")

    assert not app.exception
    assert any(
        "Could not load the decision matrix" in error.value for error in app.error
    )


def test_oversized_upload_is_rejected_without_raw_exception():
    app = _app_with_upload(b"x" * (MAX_UPLOAD_BYTES + 1), "large.csv")

    assert not app.exception
    assert any("maximum is 10 MB" in error.value for error in app.error)


def test_same_filename_replacement_resets_calculation_by_content():
    first = b"Alternative,C1,C2\nA1,1,2\nA2,2,1\n"
    replacement = b"Alternative,C1,C2\nA1,10,2\nA2,2,9\n"
    app = _app_with_upload(first, "same.csv")
    _run_aura_baseline(app)
    old_upload_fingerprint = app.session_state["prev_upload_fingerprint"]

    assert app.session_state["calculated"] is True

    app.file_uploader[0].set_value(
        ("same.csv", replacement, "text/csv")
    ).run()

    assert not app.exception
    assert app.session_state["calculated"] is False
    assert app.session_state["results_df"] is None
    assert app.session_state["calculation_fingerprint"] is None
    assert app.session_state["prev_upload_fingerprint"] != old_upload_fingerprint


def test_uppercase_csv_calculates_and_displays_every_rank_one_tie():
    tied_matrix = b"Alternative,C1\nA1,10\nA2,10\nA3,5\n"
    app = _app_with_upload(tied_matrix, "TIES.CSV")
    _run_aura_baseline(app)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["🥇 Rank-1 Alternative(s)"] == "A1, A2"
    assert metrics["Winner-to-runner-up score gap (absolute)"] == "1.1111"
    assert any("Next-ranked alternative(s): A3" in item.value for item in app.caption)


def test_sensitivity_and_comparison_run_only_on_request_and_become_stale():
    app = _app_with_upload(CRISP_MATRIX)
    _run_aura_baseline(app)

    assert app.session_state["sensitivity_result"] is None
    assert app.session_state["comparison_result"] is None

    _click_button(app, "Run Sensitivity Analysis")
    first_sensitivity_fingerprint = app.session_state["sensitivity_fingerprint"]
    sensitivity_result = app.session_state["sensitivity_result"]
    assert sensitivity_result["weight"] is not None
    assert sensitivity_result["parameter"] is not None
    assert not app.exception

    criterion = next(
        widget
        for widget in app.selectbox
        if widget.label == "Select Criterion to Vary"
    )
    criterion.set_value("C2").run()
    assert _warning_contains(app, "Sensitivity settings changed")
    assert app.session_state["sensitivity_result"] is not None

    _click_button(app, "Run Sensitivity Analysis")
    assert app.session_state["sensitivity_fingerprint"] != first_sensitivity_fingerprint
    assert not _warning_contains(app, "Sensitivity settings changed")

    _click_button(app, "Run Method Comparison")
    comparison_result = app.session_state["comparison_result"]
    assert comparison_result is not None
    assert not comparison_result["rankings"].empty

    methods = next(
        widget
        for widget in app.multiselect
        if widget.label == "Select Methods to Compare"
    )
    methods.set_value(["AURA", "TOPSIS"]).run()
    assert _warning_contains(app, "Comparison settings changed")
    assert app.session_state["comparison_result"] is not None
    assert not app.exception


def test_monte_carlo_runs_compactly_and_invalidates_changed_controls():
    app = _app_with_upload(CRISP_MATRIX)
    _run_aura_baseline(app)
    iterations = next(
        widget for widget in app.selectbox if widget.label == "Iterations"
    )
    iterations.set_value(250).run()

    assert app.session_state["monte_carlo_result"] is None
    _click_button(app, "Run Monte Carlo Simulation")

    result = app.session_state["monte_carlo_result"]
    assert not app.exception
    assert result["rank_samples"].shape == (250, 3)
    assert result["rank_samples"].dtype == np.uint16
    assert result["weight_samples"].dtype == np.float64
    assert "Raw ranks CSV" not in [button.label for button in app.download_button]
    assert "Sampled weights CSV" not in [
        button.label for button in app.download_button
    ]

    seed = next(
        widget for widget in app.number_input if widget.label == "Random seed"
    )
    seed.set_value(43).run()
    assert _warning_contains(app, "Monte Carlo settings changed")
    assert app.session_state["monte_carlo_result"] is not None

    seed.set_value(42).run()
    prepare_raw = next(
        widget
        for widget in app.checkbox
        if widget.label == "Prepare raw rank and weight CSV downloads"
    )
    prepare_raw.set_value(True).run()
    download_labels = [button.label for button in app.download_button]
    assert "Raw ranks CSV" in download_labels
    assert "Sampled weights CSV" in download_labels
    assert not app.exception
