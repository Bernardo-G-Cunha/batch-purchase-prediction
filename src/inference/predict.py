from __future__ import annotations

import mlflow

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.config import get_config
from src.common.logging import get_logger
from src.common.storage_paths import layer_path

logger = get_logger(__name__)


MODEL_FEATURES = [
    "user_views_7d",
    "user_addtocart_7d",
    "user_transactions_30d",
    "user_sessions_30d",
    "avg_session_duration_30d",
    "user_cart_items_avg_popularity_7d",
    "user_cart_top_seller_ratio_7d",
]


def configure_mlflow() -> None:
    """Configure MLflow tracking."""

    config = get_config()

    mlflow.set_tracking_uri(
        config.mlflow.tracking_uri
    )

    logger.info(
        "MLflow configured",
        extra={
            "tracking_uri": config.mlflow.tracking_uri,
        },
    )


def read_gold(spark: SparkSession) -> DataFrame:
    """Read Gold feature dataset."""

    gold_path = layer_path("gold")

    logger.info(
        "Reading gold",
        extra={"path": gold_path},
    )

    return spark.read.parquet(gold_path)


def load_model():
    """Load the production prediction model."""

    model = mlflow.pyfunc.load_model(
        "models:/RetailRocketPurchaseModel@production"
    )

    logger.info(
        "Production model loaded",
        extra={
            "model": "RetailRocketPurchaseModel",
            "alias": "production",
        },
    )

    return model


def fix_types(df):
    """Match input types expected by the production model."""

    integer_features = [
        "user_views_7d",
        "user_addtocart_7d",
        "user_transactions_30d",
        "user_sessions_30d",
    ]

    df[integer_features] = (
        df[integer_features]
        .astype("int32")
    )

    float_features = [
        "avg_session_duration_30d",
        "user_cart_items_avg_popularity_7d",
        "user_cart_top_seller_ratio_7d",
    ]

    df[float_features] = (
        df[float_features]
        .astype("float64")
    )

    return df


def prepare_inference_data(
    gold_df: DataFrame,
) -> DataFrame:
    """Prepare the latest Gold snapshot for inference."""

    inference_day = (
        gold_df
        .agg(F.max("snapshot_day"))
        .collect()[0][0]
    )

    logger.info(
        "Inference day identified",
        extra={
            "inference_day": inference_day,
        },
    )

    inference_df = (
        gold_df
        .filter(
            F.col("snapshot_day") == inference_day
        )
        .filter(
            (
                F.col("user_views_7d")
                + F.col("user_addtocart_7d")
                + F.col("user_transactions_30d")
            ) >= 5
        )
        .select(
            "visitorid",
            *MODEL_FEATURES,
        )
    )

    logger.info(
        "Inference dataset prepared",
        extra={
            "inference_day": inference_day,
        },
    )

    return inference_df


def predict(
    model,
    inference_df: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """Generate purchase probabilities."""

    prediction_input = (
        inference_df
        .select(*MODEL_FEATURES)
        .toPandas()
    )

    prediction_input = fix_types(
        prediction_input
    )

    probabilities = model.predict(
        prediction_input
    )

    prediction_df = (
        inference_df
        .select("visitorid")
        .toPandas()
    )

    prediction_df["purchase_probability"] = (
        probabilities
    )

    return spark.createDataFrame(
        prediction_df
    )


def write_predictions(
    prediction_df: DataFrame,
) -> None:
    """Write prediction dataset."""

    config = get_config()
    inference_path = config.storage.inference_path

    logger.info(
        "Writing predictions",
        extra={
            "path": inference_path,
        },
    )

    (
        prediction_df
        .write
        .mode("overwrite")
        .parquet(inference_path)
    )


def run(spark: SparkSession) -> None:

    configure_mlflow()

    model = load_model()

    gold_df = read_gold(spark)

    inference_df = prepare_inference_data(
        gold_df
    )

    prediction_df = predict(
        model=model,
        inference_df=inference_df,
        spark=spark,
    )

    write_predictions(
        prediction_df
    )


if __name__ == "__main__":

    config = get_config()

    spark = (
        SparkSession.builder
        .appName(
            f"batch-inference-{config.environment}"
        )
        .getOrCreate()
    )

    run(spark)

    spark.stop()