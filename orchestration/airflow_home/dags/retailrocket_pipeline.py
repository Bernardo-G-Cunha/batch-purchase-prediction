from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import ExternalPythonOperator


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def run_ingestion():
    from pyspark.sql import SparkSession

    from src.ingestion.events import run

    spark = (
        SparkSession.builder
        .appName("RetailRocket-Ingestion")
        .getOrCreate()
    )

    try:
        run(spark)
    finally:
        spark.stop()


def run_bronze_to_silver():
    from pyspark.sql import SparkSession

    from src.processing.bronze_to_silver import run

    spark = (
        SparkSession.builder
        .appName("RetailRocket-Bronze-to-Silver")
        .getOrCreate()
    )

    try:
        run(spark)
    finally:
        spark.stop()


def run_silver_to_gold():
    from pyspark.sql import SparkSession

    from src.processing.silver_to_gold import run

    spark = (
        SparkSession.builder
        .appName("RetailRocket-Silver-to-Gold")
        .getOrCreate()
    )

    try:
        run(spark)
    finally:
        spark.stop()


def run_inference():
    from pyspark.sql import SparkSession

    from src.inference.predict import run

    spark = (
        SparkSession.builder
        .appName("RetailRocket-Inference")
        .getOrCreate()
    )

    try:
        run(spark)
    finally:
        spark.stop()


with DAG(
    dag_id="retailrocket_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="0 0 * * 0",
    catchup=False,
) as dag:

    ingestion = ExternalPythonOperator(
        task_id="ingestion",
        python=PROJECT_PYTHON,
        python_callable=run_ingestion,
    )

    bronze_to_silver = ExternalPythonOperator(
        task_id="bronze_to_silver",
        python=PROJECT_PYTHON,
        python_callable=run_bronze_to_silver,
    )

    silver_to_gold = ExternalPythonOperator(
        task_id="silver_to_gold",
        python=PROJECT_PYTHON,
        python_callable=run_silver_to_gold,
    )

    inference = ExternalPythonOperator(
        task_id="inference",
        python=PROJECT_PYTHON,
        python_callable=run_inference,
    )

    ingestion >> bronze_to_silver >> silver_to_gold >> inference