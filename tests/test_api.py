"""Contract tests: status codes and response schema, independent of what the model predicts."""

import pytest

from app.main import MAX_CHARS, preprocess


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_schema(client):
    r = client.post("/predict", json={"text": "just a normal sentence"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"label", "score"}
    assert body["label"] in {"negative", "neutral", "positive"}
    assert 0.0 <= body["score"] <= 1.0


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": ""}, {"text": "   "}, {"text": 42}, {"text": "x" * (MAX_CHARS + 1)}],
    ids=["missing", "empty", "blank", "not-a-string", "too-long"],
)
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_accepts_max_length_text(client):
    # 2000 chars of emoji ≫ 512 tokens: must be truncated, not crash the model.
    r = client.post("/predict", json={"text": "😀" * MAX_CHARS})
    assert r.status_code == 200


def test_preprocess_masks_users_and_links():
    assert preprocess("@acme see https://t.co/xyz") == "@user see http"
