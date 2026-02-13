import pandas as pd
import pyspark.sql.functions as F
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark import SQLContext, SparkConf

import pyspark.sql.functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("test").config(
    "spark.sql.broadcastTimeout", 3600
)

date = '2026-02-12'
min_content_per_request = 5

print(f'Started for {date} !')
val_df = spark.read.parquet(f"s3://p13n-reco-offline-prod/dataset_v5/tpfy-impr-v3/daily-mtl-extracted-cms3/cd={date}/part*")
filtered_val_df = val_df.groupBy("dw_p_id", "timestamp", "flag") \
    .agg(F.countDistinct("content_id").alias("unique_content_count")) \
    .filter(F.col('flag') == 0) \
    .filter(F.col("unique_content_count") >= min_content_per_request) \
    .select("dw_p_id", "timestamp", "flag")

val_df_filtered = val_df.join(filtered_val_df, on=["dw_p_id", "timestamp", "flag"], how="inner")

groups_with_click = val_df_filtered.groupBy("dw_p_id", "timestamp") \
    .agg(F.max(F.col("click")).alias("has_click")) \
    .filter(F.col("has_click").getItem(0) == 1)\
    .select("dw_p_id", "timestamp", "has_click")

val_df_filtered_with_click = val_df_filtered.join(groups_with_click, on=["dw_p_id", "timestamp"], how="inner")

val_df_filtered_with_click = val_df_filtered_with_click.repartition(256, ["dw_p_id", "timestamp"])
val_df_filtered_with_click.write.parquet(f's3://p13n-reco-offline-prod/upload_objects/test_vedansh/daily-mtl-extracted-cms3-minimum-5-contents/cd={date}/')

print(f'Complete for {date} !')
