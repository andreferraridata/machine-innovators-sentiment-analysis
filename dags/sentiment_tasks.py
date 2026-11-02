"""Plain-Python steps of the evaluation / retraining cycle.

No Airflow import here: the DAG only orchestrates, these functions can be unit-tested and run
from a notebook. Steps exchange *small* JSON-serialisable values (file paths, metric dicts),
which is what Airflow XCom is meant for; the data itself lives on disk/object storage.
"""

import json
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path

LABELS = ("negative", "neutral", "positive")
# Public labelled stand-in for "this week's social comments". Pinned commit for reproducibility.
TWEETEVAL_COMMIT = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
TWEETEVAL = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/{TWEETEVAL_COMMIT}/datasets/sentiment/"

# Reference values measured on the full TweetEval test set (12,284 tweets) (see the delivery notebook).
BASELINE = {
    "macro_f1": 0.724,
    "label_share": {"negative": 0.379, "neutral": 0.419, "positive": 0.202},  # predicted, not true, mix
    "mean_length": 91.3,
}
MIN_MACRO_F1 = 0.65  # retrain if quality drops > ~7 points below the baseline
MAX_LABEL_SHIFT = 0.15  # total variation distance between predicted and baseline class mix
MAX_LENGTH_SHIFT = 0.5  # relative change of the mean input length


def collect_batch(run_date: str, out_dir: str, size: int = 300) -> str:
    """Fetch a labelled batch and store it as JSONL. Returns the file path.

    ponytail: rotates deterministically over TweetEval; at a client this becomes the
    social-listening export of the period plus a human-labelled sample.
    """
    texts = urllib.request.urlopen(TWEETEVAL + "test_text.txt").read().decode().splitlines()
    labels = urllib.request.urlopen(TWEETEVAL + "test_labels.txt").read().decode().split()
    start = (sum(map(ord, run_date)) * size) % (len(texts) - size)
    path = Path(out_dir) / f"batch_{run_date}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text, label in zip(texts[start : start + size], labels[start : start + size], strict=True):
            f.write(json.dumps({"text": text.strip(), "label": LABELS[int(label)]}) + "\n")
    return str(path)


def http_predictor(service_url: str) -> Callable[[str], str]:
    """Predict by calling the deployed service: evaluation measures what production serves."""

    def predict(text: str) -> str:
        req = urllib.request.Request(
            service_url.rstrip("/") + "/predict",
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.load(urllib.request.urlopen(req, timeout=30))["label"]

    return predict


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    scores = []
    for c in LABELS:
        tp = sum(t == p == c for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(p == c != t for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == c != p for t, p in zip(y_true, y_pred, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)
    return sum(scores) / len(scores)


def evaluate(batch_path: str, predict: Callable[[str], str]) -> dict:
    rows = [json.loads(line) for line in Path(batch_path).read_text(encoding="utf-8").splitlines()]
    y_true = [r["label"] for r in rows]
    y_pred = [predict(r["text"]) for r in rows]
    counts = Counter(y_pred)
    return {
        "n": len(rows),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "accuracy": round(sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(rows), 4),
        "label_share": {c: round(counts[c] / len(rows), 4) for c in LABELS},
        "mean_length": round(sum(len(r["text"]) for r in rows) / len(rows), 1),
    }


def drift_report(metrics: dict, baseline: dict = BASELINE) -> dict:
    """Decide whether to retrain. Returns the reasons, so the decision is auditable."""
    label_shift = 0.5 * sum(abs(metrics["label_share"][c] - baseline["label_share"][c]) for c in LABELS)
    length_shift = abs(metrics["mean_length"] - baseline["mean_length"]) / baseline["mean_length"]
    reasons = []
    if metrics["macro_f1"] < MIN_MACRO_F1:
        reasons.append(f"macro_f1 {metrics['macro_f1']:.3f} < {MIN_MACRO_F1} (concept drift / quality drop)")
    if label_shift > MAX_LABEL_SHIFT:
        reasons.append(f"label shift {label_shift:.3f} > {MAX_LABEL_SHIFT} (prediction drift)")
    if length_shift > MAX_LENGTH_SHIFT:
        reasons.append(f"length shift {length_shift:.2f} > {MAX_LENGTH_SHIFT} (data drift)")
    return {
        "retrain": bool(reasons),
        "reasons": reasons,
        "label_shift": round(label_shift, 4),
        "length_shift": round(length_shift, 4),
    }


def request_retraining(batch_path: str, metrics: dict, report: dict, out_dir: str) -> str:
    """Write the retraining request consumed by the training job. Returns its path.

    ponytail: a manifest file stands in for submitting the fine-tuning job
    (e.g. a GPU job on the training cluster, followed by evaluation and a gated promotion).
    """
    path = Path(out_dir) / (Path(batch_path).stem + "_retrain_request.json")
    path.write_text(json.dumps({"data": batch_path, "metrics": metrics, "report": report}, indent=2))
    return str(path)
