FROM python:3.11.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /srv

# 1) Dependencies first: this layer is rebuilt only when requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Model weights baked into the image at the pinned revision: no network at startup,
#    the image is the deployable artifact and behaves identically everywhere.
ARG MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest
ARG MODEL_REVISION=3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7
ENV MODEL_NAME=${MODEL_NAME} MODEL_REVISION=${MODEL_REVISION}
RUN python -c "from huggingface_hub import snapshot_download as d; \
d('${MODEL_NAME}', revision='${MODEL_REVISION}', allow_patterns=['*.json', '*.txt', 'pytorch_model.bin'])"
ENV HF_HUB_OFFLINE=1

# 3) Application code last: code changes don't invalidate the heavy layers above.
COPY app/ app/

RUN useradd --uid 1000 --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
