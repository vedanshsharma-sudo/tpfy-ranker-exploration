import os
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StructType,
    StructField,
    ArrayType,
    LongType,
    BooleanType,
)
import argparse
from pyspark.sql import SQLContext

from common.config.constants import DataPath
from common.spark_utils import create_spark_context, create_sql_context
from common.data.behavior.watch import Watch2
from common.config.utils import data_path, tenant_countries
from common.time_utils import (
    get_yesterday_str,
    get_diff_date,
    get_dates_str_forwards,
)
import random
from common.s3_utils import is_s3_file_exist
import bisect
from functools import reduce
from pyspark.sql import DataFrame
from common.config import TENANT
import math

MIN_WATCH_TIME = 600


def get_base_path(path):
    idx = path.index("cd=")
    if idx < 0:
        raise ValueError("no cd found")
    return path[:idx]


def print_dataframe(df):
    print("rows number: " + str(df.count()))
    df.printSchema()
    df.show(n=1000, truncate=False)


def get_bin_wrapper(max_bin_num, min_per_bin):
    def get_bin(minute_counts):
        total = sum(count for minute, count in minute_counts)
        bin_num = max(min(total // min_per_bin, max_bin_num - 1), 1)
        per_bin = total // bin_num
        bin_right_edges = [0]  # (]
        current_bin_num = 0
        for minute, count in sorted(minute_counts, key=lambda x: x[0]):
            current_bin_num += count
            if (
                minute == 1 and current_bin_num >= min_per_bin
            ) or current_bin_num >= per_bin:
                bin_right_edges.append(minute * 60)
                current_bin_num = 0
        return bin_right_edges[:max_bin_num]

    return get_bin


def load_complete_watch(sql_context, end_date, country):
    parse = Watch2.parse_watches_fn()
    data = (
        sql_context.read.option("mergeSchema", "false")
        .parquet(data_path(DataPath.S3_GROUP_WATCH_UBS, country) % end_date)
        .select("dw_p_id", "watches")
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
        .agg(F.sum("watch_len").alias("past_watch_time"))
    )
    return data


SampleSchema = StructType(
    [
        StructField("before", LongType()),
        StructField("wt_0", LongType()),
        StructField("is_sample", BooleanType()),
    ]
)


def sample_wrapper(sample_num):
    @F.udf(returnType=ArrayType(SampleSchema))
    def sample(watch_time):
        result = [[0, watch_time, False]]
        for _ in range(sample_num):  # not right, bias
            before = random.randint(0, watch_time - MIN_WATCH_TIME)
            result.append([before, watch_time - before, True])
        return result

    return sample


def run_daily_event(sql_context: SQLContext, date, country, sample_num):
    past_watches = load_complete_watch(sql_context, get_yesterday_str(date), country)
    today_watches = (
        sql_context.read.parquet(
            data_path(DataPath.S3_DAILY_WATCH_ENT_PLATFORM, country) % date
        )
        .groupBy("dw_p_id", "content_id")
        .agg(F.sum("watch_time").alias("daily_watch_time"))
    )
    today_watches = (
        today_watches.filter(F.col("daily_watch_time") >= MIN_WATCH_TIME)
        .withColumn("samples", sample_wrapper(sample_num)("daily_watch_time"))
        .withColumn("sample", F.explode(F.col("samples")))
        .select("dw_p_id", "content_id", "sample.*")
        .join(past_watches, on=["dw_p_id", "content_id"], how="left")
        .fillna({"past_watch_time": 0})
        .withColumn("watched", F.col("past_watch_time") + F.col("before"))
        .select("dw_p_id", "content_id", "watched", "wt_0", "is_sample")
        # .cache()
    )
    today_watches.repartition(16).write.mode("overwrite").parquet(
        data_path(DataPath.S3_CUMULATIVE_WATCH_EVENT, country) % date
    )


@F.udf(returnType=IntegerType())
def get_group(watched, bin_right_edge):
    if bin_right_edge is None:
        return 0
    else:
        group = bisect.bisect_left(bin_right_edge, watched)
        return min(group, len(bin_right_edge) - 1)


def watch_distribution_wrapper(days, day_range):
    @F.udf(returnType=ArrayType(IntegerType()))
    def watch_distribution(watch_time_list, event_dt, wt_0):
        visible_day = days - event_dt
        actual_day_range = min(visible_day, day_range)
        should_count = [1 for _ in range(actual_day_range)] + [
            0 for _ in range(day_range - actual_day_range)
        ]
        watch_time_filtered = [0 for _ in range(day_range)]
        watch_time_filtered[0] = wt_0
        if watch_time_list is not None:
            for watch_dt, watch_time in watch_time_list:
                if 0 < (watch_dt - event_dt) < day_range:
                    watch_time_filtered[watch_dt - event_dt] = watch_time
        return watch_time_filtered + should_count

    return watch_distribution


def run_for_country(args, countries):
    end_date, days, day_range, sample_num = (
        args.end_date,
        args.days,
        args.day_range,
        args.sample_num,
    )
    max_bin_num, min_per_bin = args.max_bin_num, args.min_per_bin
    spark_context = create_spark_context("watched watch distribution", enable_hive=True)
    sql_context = create_sql_context(spark_context)
    start_date = get_diff_date(end_date, -days + 1)
    dfs = []
    for country in countries:
        daily_event_path = data_path(DataPath.S3_CUMULATIVE_WATCH_EVENT, country)
        for i in range(days):
            dt = get_diff_date(start_date, i)
            print(f"check daily raw data {dt}")
            if (
                not is_s3_file_exist(os.path.join(daily_event_path % dt, "_SUCCESS"))
                or args.purge_event
            ):
                run_daily_event(sql_context, dt, country, sample_num)

        watch_event = (
            sql_context.read.option("basePath", get_base_path(daily_event_path))
            .parquet(daily_event_path % get_dates_str_forwards(start_date, days))
            .filter(F.col("is_sample") == False)
            .drop("is_sample")
            # .cache()
        )
        dfs.append(watch_event)
    watch_event = reduce(DataFrame.union, dfs)

    content_watch_bin = (
        watch_event.select("content_id", "watched")
        .filter(F.col("watched") > 0)
        .rdd.map(lambda t: ((t[0], math.ceil(t[1] / 60)), 1))
        .reduceByKey(lambda a, b: a + b)
        .map(lambda t: (t[0][0], (t[0][1], t[1])))
        .groupByKey()
        .mapValues(get_bin_wrapper(max_bin_num, min_per_bin))
        .toDF(["content_id", "bin_right_edge"])
        .cache()
    )
    print_dataframe(content_watch_bin)

    watches = None
    for i in range(days):
        dt = get_diff_date(start_date, i)
        print("load watch", dt)
        for country in countries:
            w = sql_context.read.parquet(
                data_path(DataPath.S3_DAILY_WATCH_ENT_PLATFORM, country) % dt
            ).withColumn("watch_dt", F.lit(i))
            if watches is None:
                watches = w
            else:
                watches = watches.union(w)
    watches = (
        watches.groupBy("dw_p_id", "content_id", "watch_dt")
        .agg(F.sum("watch_time").alias("watch_time"))
        .groupBy("dw_p_id", "content_id")
        .agg(
            F.collect_list(F.struct("watch_dt", "watch_time")).alias("watch_time_list")
        )
    )
    new_columns = [f"wt_{x}" for x in range(day_range)] + [
        f"count_{x}" for x in range(day_range)
    ]
    cumulative_watch = (
        watch_event.withColumn("event_dt", F.datediff("cd", F.lit(start_date)))
        .join(content_watch_bin, on="content_id", how="left")
        .withColumn("group", get_group("watched", "bin_right_edge"))
        .select("dw_p_id", "content_id", "event_dt", "group", "wt_0")
        .join(watches, on=["dw_p_id", "content_id"], how="left")
        .withColumn(
            "watch_distribution",
            watch_distribution_wrapper(days, day_range)(
                "watch_time_list", "event_dt", "wt_0"
            ),
        )
        .select(
            ["dw_p_id", "content_id", "group"]
            + [
                F.expr(f"watch_distribution[{i}]").alias(new_columns[i])
                for i in range(len(new_columns))
            ]
        )
        .groupBy("content_id", "group")
        .agg(
            F.count("dw_p_id").alias("num"),
            *[F.sum(x).alias(f"sum_{x}") for x in new_columns],
        )
        # .cache()
    )
    # print_dataframe(cumulative_watch)

    content_watch_bin.select(
        [F.col("content_id")]
        + [F.expr(f"bin_right_edge[{i}]").alias(f"bin_{i}") for i in range(max_bin_num)]
    ).repartition(1).write.mode("overwrite").option("header", "true").csv(
        data_path(DataPath.S3_CUMULATIVE_WATCH_BIN, TENANT) % end_date
    )
    cumulative_watch.repartition(1).write.mode("overwrite").option(
        "header", "true"
    ).csv(data_path(DataPath.S3_WATCHED_WATCH_DISTRIBUTION, TENANT) % end_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("end_date", type=str, help="end date, YYYY-mm-dd")
    parser.add_argument("days", type=int)
    parser.add_argument("--day_range", type=int, default=7)
    parser.add_argument("--max_bin_num", type=int, default=20)
    parser.add_argument("--min_per_bin", type=int, default=100)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--purge_event", action="store_true")
    parser.add_argument(
        "--countries",
        type=str,
        help="countries to run, separated with comma. "
        "default is None. fallback to region countries",
    )
    args = parser.parse_args()
    run_for_country(args, tenant_countries(args.countries))
