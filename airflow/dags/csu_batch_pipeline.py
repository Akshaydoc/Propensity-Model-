"""Batch DAG: ingest -> features -> train (4 products in parallel) -> score -> monitor -> retrain only if drift says so.
Tasks shell out to the same CLI you run by hand, so Airflow and a laptop run exactly the same code."""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
try:
    from airflow.operators.bash import BashOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator
except ImportError:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.empty import EmptyOperator
    from airflow.providers.standard.operators.python import BranchPythonOperator

PROJECT = os.environ.get("CSU_HOME", "/opt/project")
CLI = f"cd {PROJECT} && python -m csu.cli"
PRODUCTS = ["casa", "term_loan", "funded_trade", "credit_card"]


def _needs_retrain():
    files = (Path(PROJECT) / "reports" / "monitoring").glob("*_latest.json")
    flag = any(json.loads(f.read_text()).get("retrain_required") for f in files)
    return "retrain_all" if flag else "no_retrain"


with DAG("csu_batch_pipeline", start_date=datetime(2026, 1, 1), schedule="@daily", catchup=False, max_active_runs=1,
         default_args=dict(retries=1, retry_delay=timedelta(minutes=2)), tags=["csu", "batch"]) as dag:
    bootstrap = BashOperator(task_id="bootstrap_demo_data", bash_command=f"cd {PROJECT} && ([ -f data/raw/dim_customer.csv ] || python -m csu.cli generate)")
    ingest = BashOperator(task_id="ingest_batch", bash_command=f"{CLI} ingest")
    features = BashOperator(task_id="build_features", bash_command=f"{CLI} features")
    train = [BashOperator(task_id=f"train_{p}", bash_command=f"{CLI} train --product {p}") for p in PRODUCTS]
    score = BashOperator(task_id="score_and_rank", bash_command=f"{CLI} score", trigger_rule="all_success")
    monitor = BashOperator(task_id="monitor_drift", bash_command=f"{CLI} monitor")
    branch = BranchPythonOperator(task_id="drift_gate", python_callable=_needs_retrain)
    retrain = BashOperator(task_id="retrain_all", bash_command=f"{CLI} train && {CLI} score")
    skip = EmptyOperator(task_id="no_retrain")
    bootstrap >> ingest >> features >> train >> score >> monitor >> branch >> [retrain, skip]
