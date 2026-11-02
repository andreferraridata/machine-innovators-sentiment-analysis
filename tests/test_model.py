"""Behavioural tests: the pinned model must get unambiguous cases right."""

import pytest


@pytest.mark.parametrize(
    "text",
    ["I absolutely love this, best purchase I've ever made!", "Amazing support team, thank you so much!"],
)
def test_clearly_positive(client, text):
    assert client.post("/predict", json={"text": text}).json()["label"] == "positive"


@pytest.mark.parametrize(
    "text",
    ["This is the worst service ever, I want my money back.", "Terrible app, it crashes every single time."],
)
def test_clearly_negative(client, text):
    assert client.post("/predict", json={"text": text}).json()["label"] == "negative"


def test_neutral_statement(client):
    assert client.post("/predict", json={"text": "The store opens at 9am on Monday."}).json()["label"] == "neutral"
