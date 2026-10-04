from pyspark.sql import SparkSession

from src.processing.bronze_to_silver import run


spark = SparkSession.builder.appName("RetailRocket-Bronze-to-Silver").getOrCreate()

try:
    run(spark)
finally:
    spark.stop()