from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.storage_paths import layer_path

logger = get_logger(__name__)


def read_bronze(spark: SparkSession) -> DataFrame:
    """Read Bronze layer."""
    bronze_path = layer_path("bronze")
    
    logger.info("Reading bronze", extra={"path": bronze_path})

    return spark.read.parquet(bronze_path)


def calculate_quality_metrics(df: DataFrame):
    return df.agg(
        F.count("*").alias("total_rows"),
        F.sum(
            F.when(F.col("_dq_error").isNull(), 1).otherwise(0)
        ).alias("valid_rows"),
        F.sum(
            F.when(F.col("_dq_error").isNotNull(), 1).otherwise(0)
        ).alias("quarantined_rows"),
    ).collect()[0]


def validate(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Clean Bronze data."""

    duplicated_window = Window.partitionBy(
        "event_time", "visitorid", "itemid", "event", "transactionid"
    ).orderBy("event_time")

    df_with_counting = df.withColumn("_row_num", F.row_number().over(duplicated_window))

    is_duplicated = F.col("_row_num") > 1

    null_event = F.col("event").isNull()
    not_accepted_event = ~(F.col("event").isin("view", "addtocart", "transaction"))
    null_visitor_id = F.col("visitorid").isNull()
    null_item_id = F.col("itemid").isNull()
    
    transaction_without_id = (
        (F.col("event") == "transaction") &
        F.col("transactionid").isNull()
    )

    id_without_transaction = (
        F.col("transactionid").isNotNull() &
        (F.col("event") != "transaction")
    )

    null_timestamp = F.col("event_time").isNull()

    validated = df_with_counting.withColumn(
        "_dq_error",
        F.when(null_event, "null_event")
        .when(not_accepted_event, "not_accepted_event")
        .when(null_visitor_id, "null_visitor_id")
        .when(null_item_id, "null_item_id")
        .when(transaction_without_id, "transaction_without_id")
        .when(id_without_transaction, "id_without_transaction_event")
        .when(null_timestamp, "null_timestamp")
        .when(is_duplicated, "duplicated_row")
    ).drop("_row_num")
    
    return validated


def write_silver(df: DataFrame, quarantine_df: DataFrame) -> None:

    config = get_config()

    silver_path = layer_path("silver")
    quarantine_path = config.storage.quarantine_path

    logger.info("Writing silver layer", extra={"path": silver_path})
    df.write.mode("overwrite").parquet(silver_path)

    logger.info("Writing silver quarantine", extra={"path": quarantine_path})    
    quarantine_df.write.mode("overwrite").parquet(quarantine_path)


def run(spark: SparkSession) -> None:

    bronze_df = read_bronze(spark)

    validated = validate(bronze_df)

    metrics = calculate_quality_metrics(validated)

    silver_df = validated.filter(
        F.col("_dq_error").isNull()
    ).drop("_dq_error")

    quarantine_df = validated.filter(
        F.col("_dq_error").isNotNull()
    )

    write_silver(silver_df, quarantine_df)

    logger.info(
        "Silver processing complete",
        extra={
            "total_rows": metrics["total_rows"],
            "valid_rows": metrics["valid_rows"],
            "quarantined_rows": metrics["quarantined_rows"],
        },
    )


if __name__ == "__main__":
    config = get_config()

    spark = (
        SparkSession.builder
        .appName(f"bronze-to-silver-{config.environment}")
        .getOrCreate()
    )

    run(spark)

    spark.stop()