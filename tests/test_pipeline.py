"""Tests for the retraining-cycle logic (no Airflow needed)."""

import json

import sentiment_tasks as st


def test_macro_f1_perfect_and_wrong():
    y = ["negative", "neutral", "positive"]
    assert st.macro_f1(y, y) == 1.0
    assert st.macro_f1(y, ["positive", "negative", "neutral"]) == 0.0


def test_no_retrain_on_baseline():
    assert st.drift_report(st.BASELINE) == {"retrain": False, "reasons": [], "label_shift": 0.0, "length_shift": 0.0}


def test_retrain_on_quality_drop_and_drift():
    metrics = {
        "macro_f1": 0.5,
        "label_share": {"negative": 0.7, "neutral": 0.2, "positive": 0.1},
        "mean_length": 300.0,
    }
    report = st.drift_report(metrics)
    assert report["retrain"]
    assert len(report["reasons"]) == 3


def test_cycle_against_the_real_api(client, tmp_path):
    # The same functions the DAG chains, with the FastAPI app as predictor.
    batch = tmp_path / "batch.jsonl"
    rows = [
        {"text": "I love this so much!", "label": "positive"},
        {"text": "This is awful, I hate it.", "label": "negative"},
        {"text": "The train leaves at 10.", "label": "neutral"},
    ]
    batch.write_text("\n".join(json.dumps(r) for r in rows))

    metrics = st.evaluate(str(batch), lambda t: client.post("/predict", json={"text": t}).json()["label"])
    assert metrics["n"] == 3 and metrics["macro_f1"] == 1.0

    report = st.drift_report(metrics)
    request = st.request_retraining(str(batch), metrics, report, str(tmp_path))
    assert json.loads(open(request).read())["metrics"] == metrics
