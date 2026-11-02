"""Sentiment inference service: FastAPI wrapper around cardiffnlp/twitter-roberta-base-sentiment-latest."""

import os
import re
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_NAME = os.getenv("MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")
# Pinned commit of the model repo on the Hub: same weights in every environment.
MODEL_REVISION = os.getenv("MODEL_REVISION", "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7")
MAX_CHARS = 2000  # ~ well above a tweet; the tokenizer truncates at 512 tokens anyway


def preprocess(text: str) -> str:
    """Same normalisation used to train the model (see the model card)."""
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"http\S+", "http", text)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS, examples=["I love this product!"])


class PredictResponse(BaseModel):
    label: Literal["negative", "neutral", "positive"]
    score: float = Field(ge=0, le=1)


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Loaded once at startup, never per request.
    state["classifier"] = pipeline("sentiment-analysis", model=MODEL_NAME, revision=MODEL_REVISION)
    yield
    state.clear()


app = FastAPI(title="Sentiment API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "revision": MODEL_REVISION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be blank")
    # The tokenizer config has no model_max_length: without an explicit cap a long
    # text would overflow RoBERTa's 512 position embeddings and crash.
    result = state["classifier"](preprocess(req.text), truncation=True, max_length=512)[0]
    return PredictResponse(label=result["label"], score=result["score"])
