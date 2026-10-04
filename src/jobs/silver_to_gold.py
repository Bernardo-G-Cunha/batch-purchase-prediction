from pyspark.sql import SparkSession

from src.processing.silver_to_gold import run


spark = SparkSession.builder.appName("RetailRocket-Silver-to-Gold").getOrCreate()

try:
    run(spark)
finally:
    spark.stop()