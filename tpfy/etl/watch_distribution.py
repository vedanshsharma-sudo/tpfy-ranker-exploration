from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StructType,
    StructField,
    StringType,
    ArrayType,
)
import argparse

from common.config.constants import DataPath, EMPTY_USER_DW_PID
from common.spark_utils import create_spark_context, create_sql_context, set_partitions
from common.time_utils import get_dates_str_backwards, timestamp
from common.data.behavior.watch import Watch2
from common.config.utils import data_path, get_config, tenant_countries


def load_complete_watch(sql_context, end_date, country):
    parse = Watch2.parse_watches_fn()
    data = (
        sql_context.read.option("mergeSchema", "false")
        .parquet(data_path(DataPath.S3_GROUP_WATCH_UBS, country) % end_date)
        .rdd.flatMap(lambda t: [[t[0], *watch] for watch in parse(t[1])])
        .toDF(
            [
                "dw_p_id",
                "content_id",
                "language",
                "watch_len",
                "first_watch",
                "last_watch",
            ]
        )
        .groupBy("dw_p_id", "content_id")
        .agg(F.min("first_watch").alias("complete_first_watch_ts"))
    )
    return data


def load_daily_watches(sql_context, platform_path, end_date, days, abnormal_users):
    dates_str = get_dates_str_backwards(end_date, days)
    data = sql_context.read.option("mergeSchema", "false").parquet(
        platform_path % dates_str
    )
    return (
        data.select(
            [
                "dw_p_id",
                "content_id",
                "language",
                "watch_time",
                "first_watch_ts",
                "last_watch_ts",
                "platform_id",
            ]
        )
        .filter(F.col("dw_p_id").isNotNull())
        .filter(F.col("dw_p_id") != "")
        .join(abnormal_users, on="dw_p_id", how="left_anti")
        .filter(F.col("dw_p_id") != EMPTY_USER_DW_PID)
        .filter(F.col("content_id").isNotNull())
        .withColumn("watch_time", F.col("watch_time").cast(IntegerType()))
        .withColumn("first_watch_ts", F.col("first_watch_ts").cast(IntegerType()))
        .withColumn("last_watch_ts", F.col("last_watch_ts").cast(IntegerType()))
    )


def watch_aggregators():
    return [
        F.collect_list("watch_time").alias("watch_time_list"),
        F.min("first_watch_ts").alias("first_watch_ts"),
        F.collect_list("last_watch_ts").alias("watch_ts_list"),
    ]


def watch_distribution_wrapper(now_ts, day_range):
    @F.udf(returnType=ArrayType(IntegerType()))
    def watch_distribution(watch_time_list, first_watch_ts, watch_ts_list):
        should_count = [0 for _ in range(day_range + 1)]
        complete_days = (
            now_ts - first_watch_ts
        ) // 86400  # drop last day because that day may not complete
        complete_days = min(complete_days, day_range + 1)
        if complete_days >= 1:
            should_count[:complete_days] = [1] * complete_days
        wts = [0 for _ in range(day_range + 1)]
        for i in range(len(watch_time_list)):
            days = (watch_ts_list[i] - first_watch_ts) // 86400
            if days < 0 or days > day_range:
                continue
            if should_count[days] == 0:
                continue
            wts[days] = wts[days] + watch_time_list[i]

        return wts + should_count

    return watch_distribution


def run_for_country(args, country):
    end_date, days, day_range = args.end_date, args.days, args.day_range
    now_ts = timestamp(end_date) + 86400

    spark_context = create_spark_context("User Watch from %s" % end_date)
    sql_context = create_sql_context(spark_context)
    set_partitions(sql_context, 512, adjust=True)

    abnormal_users = sql_context.read.option("mergeSchema", "false").csv(
        (data_path(DataPath.S3_DAILY_ABNORMAL_USERS, country) % "latest").replace(
            "cd=", ""
        ),
        schema=StructType([StructField("dw_p_id", StringType(), True)]),
    )

    new_columns = [f"wt_{x}" for x in range(day_range + 1)] + [
        f"count_{x}" for x in range(day_range + 1)
    ]
    watch_distribution = watch_distribution_wrapper(now_ts, day_range)
    watch_complete = load_complete_watch(sql_context, end_date, country)

    ent_watches = (
        load_daily_watches(
            sql_context,
            data_path(DataPath.S3_DAILY_WATCH_ENT_PLATFORM, country),
            end_date,
            days,
            abnormal_users,
        )
        .groupBy("dw_p_id", "content_id")
        .agg(*watch_aggregators())
        .join(watch_complete, on=["dw_p_id", "content_id"], how="inner")
        .filter(
            F.col("first_watch_ts") <= F.col("complete_first_watch_ts")
        )  # filter out not first watch content
        .withColumn(
            "watch_distribution",
            watch_distribution("watch_time_list", "first_watch_ts", "watch_ts_list"),
        )
        .select(
            ["dw_p_id", "content_id"]
            + [
                F.expr(f"watch_distribution[{i}]").alias(new_columns[i])
                for i in range(len(new_columns))
            ]
        )
        .filter(F.col("wt_0") >= 600)
        .groupBy("content_id")
        .agg(
            F.count("dw_p_id").alias("num"),
            *[F.sum(x).alias(f"sum_{x}") for x in new_columns],
        )
    )

    ent_watches.repartition(1).write.mode("overwrite").option("header", "true").csv(
        data_path(DataPath.S3_WATCH_DISTRIBUTION, country) % end_date
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("end_date", type=str, help="end date, YYYY-mm-dd")
    parser.add_argument("days", type=int)
    parser.add_argument("--day_range", type=int, default=30)
    parser.add_argument(
        "--countries",
        type=str,
        help="countries to run, separated with comma. "
        "default is None. fallback to region countries",
    )
    args = parser.parse_args()
    for country in tenant_countries(args.countries):
        run_for_country(args, country)
