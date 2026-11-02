# machine-innovators-sentiment-analysis

Social sentiment monitoring service (positive / neutral / negative) built on
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt
uvicorn app.main:app --port 8000
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text": "I love it"}'
```

`requirements.txt` is the fully pinned lock compiled from `requirements.in` (see the header of that file).
