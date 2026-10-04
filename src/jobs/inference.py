from pyspark.sql import SparkSession

from src.inference.predict import run


spark = SparkSession.builder.appName("RetailRocket-Inference").getOrCreate()

try:
    run(spark)
finally:
    spark.stop()