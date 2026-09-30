from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.storage_paths import layer_path

logger = get_logger(__name__)

EPOCH_DATE = "1970-01-01"


def read_silver(spark: SparkSession) -> DataFrame:
    """Read Silver layer."""

    silver_path = layer_path("silver")

    logger.info(
        "Reading silver",
        extra={"path": silver_path},
    )

    return spark.read.parquet(silver_path)


def add_snapshot_day(df: DataFrame) -> DataFrame:
    """Add the integer day index used by feature windows."""

    return df.withColumn(
        "snapshot_day",
        F.datediff(
            F.to_date("event_time"),
            F.lit(EPOCH_DATE),
        ),
    )


def build_snapshot_dates(df: DataFrame) -> DataFrame:
    """Build visitor-day snapshots including the inference day."""

    reference_snapshot_day = (
        df
        .select(
            F.max("snapshot_day").alias("reference_snapshot_day")
        )
        .collect()[0]["reference_snapshot_day"]
    )

    if reference_snapshot_day is None:
        raise ValueError("Silver dataset is empty.")

    inference_snapshot_day = reference_snapshot_day + 1

    historical_snapshots = (
        df
        .select(
            "visitorid",
            "snapshot_day",
        )
        .distinct()
    )

    inference_snapshots = (
        df
        .select("visitorid")
        .distinct()
        .withColumn(
            "snapshot_day",
            F.lit(inference_snapshot_day).cast("int"),
        )
    )

    return (
        historical_snapshots
        .unionByName(inference_snapshots)
        .distinct()
    )


def build_user_behavior(
    silver_df: DataFrame,
    snapshot_data: DataFrame,
) -> DataFrame:
    """Build rolling user behavior features from Silver events."""

    # One row per visitor/day.
    daily = (
        silver_df
        .groupBy("visitorid", "snapshot_day")
        .agg(
            F.sum(
                F.when(
                    F.col("event") == "view",
                    1,
                ).otherwise(0)
            ).alias("views"),
            F.sum(
                F.when(
                    F.col("event") == "addtocart",
                    1,
                ).otherwise(0)
            ).alias("addtocarts"),
            F.sum(
                F.when(
                    F.col("event") == "transaction",
                    1,
                ).otherwise(0)
            ).alias("transactions"),
        )
        .join(
            snapshot_data,
            on=["visitorid", "snapshot_day"],
            how="right",
        )
        .withColumn(
            "views",
            F.coalesce(
                F.col("views"),
                F.lit(0),
            ),
        )
        .withColumn(
            "addtocarts",
            F.coalesce(
                F.col("addtocarts"),
                F.lit(0),
            ),
        )
        .withColumn(
            "transactions",
            F.coalesce(
                F.col("transactions"),
                F.lit(0),
            ),
        )
    )

    window_7d = (
        Window
        .partitionBy("visitorid")
        .orderBy("snapshot_day")
        .rangeBetween(-7, -1)
    )

    window_30d = (
        Window
        .partitionBy("visitorid")
        .orderBy("snapshot_day")
        .rangeBetween(-30, -1)
    )

    features = (
        daily
        .withColumn(
            "user_views_7d",
            F.coalesce(
                F.sum("views").over(window_7d),
                F.lit(0),
            ),
        )
        .withColumn(
            "user_addtocart_7d",
            F.coalesce(
                F.sum("addtocarts").over(window_7d),
                F.lit(0),
            ),
        )
        .withColumn(
            "user_transactions_30d",
            F.coalesce(
                F.sum("transactions").over(window_30d),
                F.lit(0),
            ),
        )
    )

    return features.select(
        "visitorid",
        "snapshot_day",
        "user_views_7d",
        "user_addtocart_7d",
        "user_transactions_30d",
    )


def add_session_boundaries(df: DataFrame) -> DataFrame:
    """Identify where a new session starts."""

    window = (
        Window
        .partitionBy("visitorid")
        .orderBy("event_time")
    )

    return (
        df
        .withColumn(
            "previous_event_time",
            F.lag("event_time").over(window),
        )
        .withColumn(
            "time_diff",
            F.col("event_time")
            - F.col("previous_event_time"),
        )
        .withColumn(
            "new_session",
            F.when(
                F.col("time_diff").isNull()
                | (
                    F.col("time_diff")
                    > F.expr("INTERVAL 30 MINUTES")
                ),
                1,
            ).otherwise(0),
        )
    )


def add_session_id(df: DataFrame) -> DataFrame:
    """Assign a sequential session ID to each visitor's events."""

    session_window = (
        Window
        .partitionBy("visitorid")
        .orderBy("event_time")
        .rowsBetween(
            Window.unboundedPreceding,
            Window.currentRow,
        )
    )

    return df.withColumn(
        "session_id",
        F.sum("new_session").over(session_window),
    )


def aggregate_sessions(df: DataFrame) -> DataFrame:
    """Aggregate events into individual sessions."""

    sessions = (
        df
        .groupBy(
            "visitorid",
            "session_id",
        )
        .agg(
            F.min("event_time").alias("session_start"),
            F.max("event_time").alias("session_end"),
        )
    )

    return sessions.withColumn(
        "session_duration_min",
        F.timestamp_diff(
            "SECOND",
            F.col("session_start"),
            F.col("session_end"),
        ) / 60.0,
    )


def build_daily_session_stats(
    sessions: DataFrame,
) -> DataFrame:
    """Aggregate session statistics by visitor and session start day."""

    return (
        sessions
        .withColumn(
            "snapshot_day",
            F.datediff(
                F.to_date("session_start"),
                F.lit(EPOCH_DATE),
            ),
        )
        .groupBy(
            "visitorid",
            "snapshot_day",
        )
        .agg(
            F.sum("session_duration_min")
            .alias("total_duration"),
            F.count("session_id")
            .alias("sessions"),
        )
    )


def align_session_stats_with_snapshots(
    daily_session_stats: DataFrame,
    snapshot_data: DataFrame,
) -> DataFrame:
    """Add zero-valued session statistics for snapshot days without sessions."""

    snapshots = (
        snapshot_data
        .select(
            "visitorid",
            "snapshot_day",
        )
        .dropDuplicates()
    )

    return (
        daily_session_stats
        .join(
            snapshots,
            on=["visitorid", "snapshot_day"],
            how="right",
        )
        .withColumn(
            "sessions",
            F.coalesce(
                F.col("sessions"),
                F.lit(0),
            ),
        )
        .withColumn(
            "total_duration",
            F.coalesce(
                F.col("total_duration"),
                F.lit(0.0),
            ),
        )
    )


def build_rolling_session_features(
    daily_session_stats: DataFrame,
) -> DataFrame:
    """Build 30-day rolling session features."""

    window_30d = (
        Window
        .partitionBy("visitorid")
        .orderBy("snapshot_day")
        .rangeBetween(-30, -1)
    )

    features = (
        daily_session_stats
        .withColumn(
            "user_sessions_30d",
            F.coalesce(
                F.sum("sessions").over(window_30d),
                F.lit(0),
            ),
        )
        .withColumn(
            "total_duration_30d",
            F.coalesce(
                F.sum("total_duration").over(window_30d),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "avg_session_duration_30d",
            F.when(
                F.col("user_sessions_30d") > 0,
                F.col("total_duration_30d")
                / F.col("user_sessions_30d"),
            ).otherwise(0.0),
        )
    )

    return features.select(
        "visitorid",
        "snapshot_day",
        "user_sessions_30d",
        "avg_session_duration_30d",
    )


def build_session_features(
    silver_df: DataFrame,
    snapshot_data: DataFrame,
) -> DataFrame:
    """Build rolling session features from Silver events."""

    df = add_session_boundaries(silver_df)
    df = add_session_id(df)

    sessions = aggregate_sessions(df)

    daily_session_stats = build_daily_session_stats(
        sessions
    )

    daily_session_stats = align_session_stats_with_snapshots(
        daily_session_stats,
        snapshot_data,
    )

    return build_rolling_session_features(
        daily_session_stats
    )


def build_item_popularity_features(
    df: DataFrame,
    snapshot_data: DataFrame,
) -> DataFrame:
    """Build cart item popularity and top-seller features."""

    item_daily_transactions = (
        df
        .filter(F.col("event") == "transaction")
        .select(
            "itemid",
            "snapshot_day",
        )
        .groupBy(
            "itemid",
            "snapshot_day",
        )
        .agg(
            F.count("*").alias("transactions")
        )
    )

    base = (
        df
        .select("itemid")
        .distinct()
        .crossJoin(
            df
            .select("snapshot_day")
            .distinct()
        )
    )

    item_daily_transactions = (
        base
        .join(
            item_daily_transactions,
            on=["itemid", "snapshot_day"],
            how="left",
        )
        .withColumn(
            "transactions",
            F.coalesce(
                F.col("transactions"),
                F.lit(0),
            ),
        )
    )

    item_window_30d = (
        Window
        .partitionBy("itemid")
        .orderBy("snapshot_day")
        .rangeBetween(-30, -1)
    )

    item_daily_transactions = (
        item_daily_transactions
        .withColumn(
            "item_transactions_30d",
            F.coalesce(
                F.sum("transactions").over(item_window_30d),
                F.lit(0),
            ),
        )
    )

    top_10_threshold = (
        item_daily_transactions
        .groupBy("snapshot_day")
        .agg(
            F.percentile_approx(
                "item_transactions_30d",
                0.90,
            ).alias("top_10_threshold")
        )
    )

    item_daily_transactions = (
        item_daily_transactions
        .join(
            top_10_threshold,
            on="snapshot_day",
            how="left",
        )
        .withColumn(
            "is_top_seller",
            F.when(
                F.col("item_transactions_30d")
                >= F.col("top_10_threshold"),
                1,
            ).otherwise(0),
        )
    )

    cart_events = (
        df
        .filter(F.col("event") == "addtocart")
        .select(
            "visitorid",
            "itemid",
            "snapshot_day",
        )
        .join(
            item_daily_transactions.select(
                "itemid",
                "snapshot_day",
                "item_transactions_30d",
                "is_top_seller",
            ),
            on=["itemid", "snapshot_day"],
            how="left",
        )
    )

    daily_cart_stats = (
        cart_events
        .groupBy(
            "visitorid",
            "snapshot_day",
        )
        .agg(
            F.sum("item_transactions_30d")
            .alias("cart_item_popularity_sum"),
            F.count("*")
            .alias("cart_count"),
            F.sum("is_top_seller")
            .alias("top_seller_cart_count"),
        )
    )

    daily_cart_stats = (
        daily_cart_stats
        .join(
            snapshot_data,
            on=["visitorid", "snapshot_day"],
            how="right",
        )
        .withColumn(
            "cart_item_popularity_sum",
            F.coalesce(
                F.col("cart_item_popularity_sum"),
                F.lit(0),
            ),
        )
        .withColumn(
            "cart_count",
            F.coalesce(
                F.col("cart_count"),
                F.lit(0),
            ),
        )
        .withColumn(
            "top_seller_cart_count",
            F.coalesce(
                F.col("top_seller_cart_count"),
                F.lit(0),
            ),
        )
    )

    user_window_7d = (
        Window
        .partitionBy("visitorid")
        .orderBy("snapshot_day")
        .rangeBetween(-7, -1)
    )

    features = (
        daily_cart_stats
        .withColumn(
            "cart_item_popularity_sum_7d",
            F.coalesce(
                F.sum(
                    "cart_item_popularity_sum"
                ).over(user_window_7d),
                F.lit(0),
            ),
        )
        .withColumn(
            "cart_count_7d",
            F.coalesce(
                F.sum("cart_count").over(user_window_7d),
                F.lit(0),
            ),
        )
        .withColumn(
            "top_seller_cart_count_7d",
            F.coalesce(
                F.sum(
                    "top_seller_cart_count"
                ).over(user_window_7d),
                F.lit(0),
            ),
        )
        .withColumn(
            "user_cart_items_avg_popularity_7d",
            F.coalesce(
                F.try_divide(
                    "cart_item_popularity_sum_7d",
                    "cart_count_7d",
                ),
                F.lit(0),
            ),
        )
        .withColumn(
            "user_cart_top_seller_ratio_7d",
            F.coalesce(
                F.try_divide(
                    "top_seller_cart_count_7d",
                    "cart_count_7d",
                ),
                F.lit(0),
            ),
        )
    )

    return features.select(
        "visitorid",
        "snapshot_day",
        "user_cart_items_avg_popularity_7d",
        "user_cart_top_seller_ratio_7d",
    )


def write_gold(df: DataFrame) -> None:
    path = layer_path("gold")

    logger.info(
        "Writing gold layer",
        extra={"path": path},
    )

    df.write.mode("overwrite").parquet(path)


def run(spark: SparkSession) -> None:

    silver_df = read_silver(spark)

    silver_df = add_snapshot_day(silver_df)

    snapshots_df = build_snapshot_dates(silver_df)

    event_features = build_user_behavior(
        silver_df,
        snapshots_df,
    )

    session_features = build_session_features(
        silver_df,
        snapshots_df,
    )

    popularity_features = build_item_popularity_features(
        silver_df,
        snapshots_df,
    )

    gold_df = (
        event_features
        .join(
            session_features,
            on=["visitorid", "snapshot_day"],
            how="left",
        )
        .join(
            popularity_features,
            on=["visitorid", "snapshot_day"],
            how="left",
        )
    )

    write_gold(gold_df)

    logger.info(
        "Gold processing complete",
        extra={"row_count": gold_df.count()},
    )


if __name__ == "__main__":
    config = get_config()

    spark = (
        SparkSession.builder
        .appName(
            f"silver-to-gold-{config.environment}"
        )
        .getOrCreate()
    )

    run(spark)

    spark.stop()