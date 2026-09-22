from __future__ import annotations

from datetime import timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.storage_paths import layer_path

logger = get_logger(__name__)


def read_raw_events(spark: SparkSession) -> DataFrame:
    """Read RetailRocket event data from Parquet files."""

    config = get_config()
    path = config.storage.raw_path

    logger.info("Reading raw events", extra={"path": path})

    return spark.read.parquet(path)


def get_reference_date(df: DataFrame):
    """Get the latest event date available in the raw dataset."""

    return (
        df
        .select(F.max(F.to_date("timestamp")).alias("reference_date"))
        .collect()[0]["reference_date"]
    )


def filter_feature_window(
    df: DataFrame,
    reference_date,
) -> DataFrame:
    """Keep the 31-day window required for feature generation."""

    start_date = reference_date - timedelta(days=30)

    logger.info(
        "Filtering ingestion window",
        extra={
            "start_date": str(start_date),
            "reference_date": str(reference_date),
        },
    )

    return df.filter(
        F.to_date("timestamp").between(
            F.lit(start_date),
            F.lit(reference_date),
        )
    )


def cast_to_bronze(df: DataFrame) -> DataFrame:
    """Normalize the raw schema into the Bronze schema.

    No business rules are applied here.
    """

    return df.select(
        F.col("timestamp").alias("event_time"),
        F.col("visitorid").cast("long"),
        F.col("event"),
        F.col("itemid").cast("long"),
        F.col("transactionid").cast("long"),
    )


def write_bronze(df: DataFrame) -> None:
    path = layer_path("bronze")

    logger.info("Writing bronze layer", extra={"path": path})

    df.write.mode("overwrite").parquet(path)


def run(spark: SparkSession) -> None:
    raw_df = read_raw_events(spark)

    reference_date = get_reference_date(raw_df)

    filtered_df = filter_feature_window(
        raw_df,
        reference_date,
    )

    bronze_df = cast_to_bronze(filtered_df)

    write_bronze(bronze_df)

    logger.info(
        "Bronze ingestion complete",
        extra={"row_count": bronze_df.count()},
    )


if __name__ == "__main__":
    config = get_config()

    spark = (
        SparkSession.builder
        .appName(f"load-events-{config.environment}")
        .getOrCreate()
    )

    run(spark)

    spark.stop()