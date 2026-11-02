"""Weekly cycle: collect labelled data -> evaluate the served model -> decide -> (re)train.

    collect_data -> evaluate_model -> check_drift -> decide ─┬─> trigger_retraining
                                                             └─> no_retraining
Only paths and small dicts travel through XCom; the batch itself stays on disk.
"""

import os
from datetime import datetime

import sentiment_tasks as st
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator

DATA_DIR = os.getenv("SENTIMENT_DATA_DIR", "/opt/airflow/data/sentiment")
SERVICE_URL = os.getenv("SENTIMENT_API_URL", "http://api:8000")


@dag(
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["sentiment", "mlops"],
    default_args={"retries": 2},
)
def sentiment_retraining():
    @task
    def collect_data(ds=None) -> str:
        return st.collect_batch(ds, DATA_DIR)

    @task
    def evaluate_model(batch_path: str) -> dict:
        return st.evaluate(batch_path, st.http_predictor(SERVICE_URL))

    @task
    def check_drift(metrics: dict) -> dict:
        return st.drift_report(metrics)

    @task.branch
    def decide(report: dict) -> str:
        return "trigger_retraining" if report["retrain"] else "no_retraining"

    @task
    def trigger_retraining(batch_path: str, metrics: dict, report: dict) -> str:
        return st.request_retraining(batch_path, metrics, report, DATA_DIR)

    batch = collect_data()
    metrics = evaluate_model(batch)
    report = check_drift(metrics)
    decide(report) >> [trigger_retraining(batch, metrics, report), EmptyOperator(task_id="no_retraining")]


sentiment_retraining()
