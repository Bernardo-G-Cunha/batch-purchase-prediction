from pyspark.sql import SparkSession

from src.ingestion.events import run


spark = SparkSession.builder.appName("RetailRocket-Ingestion").getOrCreate()

try:
    run(spark)
finally:
    spark.stop()