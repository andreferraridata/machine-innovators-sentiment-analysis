# machine-innovators-sentiment-analysis

Servizio di monitoraggio del sentiment sui social (positivo / neutro / negativo) basato su
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
con test, container, CI/CD, monitoraggio e orchestrazione.

La documentazione delle scelte, eseguita e con gli output, è nel notebook
[`notebooks/delivery.ipynb`](notebooks/delivery.ipynb) (versione Colab linkata in cima al notebook).

## Struttura

| Percorso | Cosa contiene |
|---|---|
| `app/main.py` | API FastAPI: `POST /predict`, `GET /health`, `GET /metrics`. Modello caricato una volta all'avvio. |
| `tests/` | pytest: contratto API, comportamento del modello, logica del ciclo di retraining. |
| `Dockerfile` | `python:3.11.11-slim`, layer dipendenze separato, pesi del modello inclusi alla revisione fissata. |
| `.github/workflows/ci-cd.yml` | CI (ruff + pytest + verifica DAG) → CD (build, smoke test container e stack di monitoraggio, push su GHCR, deploy opzionale su HF Space). |
| `docker-compose.yml`, `monitoring/` | API + Prometheus + Grafana con datasource, dashboard e alert come codice. |
| `dags/` | DAG Airflow del ciclo raccolta → valutazione → drift → retraining; logica in `sentiment_tasks.py`. |
| `requirements.in` / `requirements.txt` | Dipendenze dirette / lock completo con versioni fissate (compilato con `uv pip compile`). |

## Avvio rapido

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest && ruff check . && ruff format --check .
uvicorn app.main:app --port 8000
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text": "I love it"}'
```

Container e monitoraggio:

```bash
docker build -t sentiment-api:ci .            # oppure: docker pull ghcr.io/andreferraridata/machine-innovators-sentiment-analysis:latest
docker run -p 8000:8000 sentiment-api:ci
docker compose up -d --build                  # API :8000 · Prometheus :9090 · Grafana :3000 (admin/admin)
```

## Deploy su Hugging Face Space (opzionale)

Il job `deploy-hf-space` si attiva solo se nella repo sono configurati il secret `HF_TOKEN`
(token HF con permesso *write*) e la variabile `HF_SPACE` (es. `utente/sentiment-api`).
Carica `Dockerfile`, `requirements.txt` e `app/` in uno Space Docker (porta 8000, utente uid 1000).
