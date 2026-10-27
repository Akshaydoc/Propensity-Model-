"""Micro-batch streaming DAG (every 5 min): pull new event files -> validate/append -> stream drift check -> maybe trigger the batch DAG.
`produce_demo_feed` stands in for the upstream system that would normally write the files."""
import json
import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
try:
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator
except ImportError:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.python import ShortCircuitOperator
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT = os.environ.get("CSU_HOME", "/opt/project")
CLI = f"cd {PROJECT} && python -m csu.cli"


def _stream_alert():
    f = Path(PROJECT) / "reports" / "monitoring" / "stream_status.json"
    return bool(f.exists() and json.loads(f.read_text()).get("alert"))


with DAG("csu_stream_pipeline", start_date=datetime(2026, 1, 1), schedule="*/5 * * * *", catchup=False, max_active_runs=1, tags=["csu", "stream"]) as dag:
    produce = BashOperator(task_id="produce_demo_feed", bash_command=f"{CLI} stream-produce --events 400")
    consume = BashOperator(task_id="consume_micro_batch", bash_command=f"{CLI} stream-consume")
    gate = ShortCircuitOperator(task_id="stream_drift_alert", python_callable=_stream_alert)
    trigger = TriggerDagRunOperator(task_id="trigger_batch_refresh", trigger_dag_id="csu_batch_pipeline")
    produce >> consume >> gate >> trigger
