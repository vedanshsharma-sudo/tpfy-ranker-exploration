import argparse
from collections import defaultdict
import os
import pickle

import pyspark.sql.functions as F
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import col
from pyspark.sql.types import BooleanType, FloatType, IntegerType
from pyspark.sql.window import Window

from common.config.utils import tenant_countries
from common.data.behavior.watch import Watch2
from common.s3_utils import (
    is_s3_file_exist,
    download_single_file,
    is_s3_path_success,
    upload_file,
    remove_s3_folder,
)
from common.spark_utils import *
from common.time_utils import (
    get_yesterday_str,
    get_dates_list_backwards,
    get_diff_date,
    get_dates_str_backwards,
)
from model.emr_utils import dict_to_spark_row
from tpfy.common import TpfyDataPath
from tpfy.cum_watch.cum_watch_schema import CumWatchDataSchema
from tpfy.cum_watch.metadata import CumWatchMetadata, CmsData
from tpfy.cum_watch.feature import extract_feature_using_fd
from tpfy.etl.lib import load_tpfy_predict_events


@F.udf(returnType=BooleanType())
def is_new_watch(content_id, watched_content_ids, raw_past_watches):
    if watched_content_ids is not None and len(watched_content_ids) > 0:
        for cid in watched_content_ids:
            if cid == content_id:
                return False
    if raw_past_watches is not None and raw_past_watches != "":
        watches = Watch2.parse_watches_fn()(raw_past_watches)
        for watch in watches:
            if watch.content_id == content_id:
                return False
    return True


@F.udf(returnType=IntegerType())
def get_past_watch_time(content_id, watched_content_ids, raw_past_watches):
    is_new = True
    watch_time = 0
    if watched_content_ids is not None and len(watched_content_ids) > 0:
        for cid in watched_content_ids:
            if cid == content_id:
                is_new = False
    if raw_past_watches is not None and raw_past_watches != "":
        watch_time_agg = defaultdict(int)
        watches = Watch2.parse_watches_fn()(raw_past_watches)
        for watch in watches:
            watch_time_agg[watch.content_id] += watch.watch_len
        if content_id in watch_time_agg:
            watch_time = watch_time_agg[content_id]
            is_new = False
    if is_new:
        return 0
    elif watch_time > 0:
        return int(watch_time)
    else:
        return -1


def run_discovery(spark, date, country, watched_sample_rate, discovery_basepath):
    dest_path = data_path(discovery_basepath, country) % date
    if is_s3_path_success(dest_path):
        print(f"discovery {date} {country} already exist; path")
        return

    print(f"run discovery dataset for {country} on {date}; dest {dest_path}")

    latest_content_ids = spark.read.parquet(
        data_path(DataPath.S3_LATEST_CONTENT_ID, country) % get_yesterday_str(date)
    ).select(
        "dw_p_id",
        col("content_ids").getField("content_id").alias("watched_content_ids"),
    )

    past_watches = spark.read.parquet(
        data_path(DataPath.S3_GROUP_WATCH_UBS, country) % get_yesterday_str(date)
    ).select("dw_p_id", col("watches").alias("raw_past_watches"))

    today_watches = (
        spark.read.parquet(
            data_path(DataPath.S3_DAILY_WATCH_ENT_PLATFORM, country) % date
        )
        .groupBy("dw_p_id", "content_id", "language")
        .agg(
            F.sum("watch_time").alias("watch_time"),
            F.min("first_watch_ts").alias("timestamp"),
        )
        .withColumn(
            "wt_rank",
            F.row_number().over(
                Window.partitionBy("dw_p_id", "content_id").orderBy(
                    col("watch_time").desc()
                )
            ),
        )
        .filter(col("wt_rank") == 1)
        .filter(F.col("watch_time") > 600)
        .drop(col("wt_rank"))
    )

    new_watches = (
        today_watches.join(latest_content_ids, on="dw_p_id", how="left")
        .join(past_watches, on="dw_p_id", how="left")
        .withColumn(
            "past_wt",
            get_past_watch_time(
                "content_id", "watched_content_ids", "raw_past_watches"
            ),
        )
        .filter(
            (F.col("past_wt") == 0)
            | ((F.col("past_wt") > 0) & (F.rand() < watched_sample_rate))
        )
        .drop("watched_content_ids")
    )

    new_watches.persist()
    print("stats")
    new_watches.withColumn(
        "watched", (F.col("past_wt") > 0).cast(IntegerType())
    ).groupBy("watched").agg(F.count(F.lit(1)).alias("cnt")).show()

    base_dataset = new_watches
    base_dataset.repartition(32, "dw_p_id").write.mode("overwrite").parquet(dest_path)

    new_watches.unpersist()


def load_metadata(sql_context, date, tenant):
    filename = f"cms_{date}.pkl"
    metadata_template = TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_METADATA_CMS3
    s3_cache_path = data_path(metadata_template, tenant) % filename
    countries = tenant_countries()
    if is_s3_file_exist(s3_cache_path):
        print("hit cms data cache", s3_cache_path)
        download_single_file(s3_cache_path, filename, from_folder=False)
        with open(filename, "rb") as f:
            cms_data = pickle.load(f)
        os.remove(filename)
    else:
        print("load metadata")
        cms_data = CmsData(countries, sql_context)
        with open(filename, "wb") as f:
            print("dump cms data", s3_cache_path)
            pickle.dump(cms_data, f)
        upload_file(s3_cache_path, filename)
        os.remove(filename)

    metadata = CumWatchMetadata(date, countries, sql_context, cms_data)

    return metadata


def get_labels(spark, date, country, cum_days):
    print("label agg dates", get_dates_list_backwards(date, cum_days))
    label = (
        spark.read.parquet(
            data_path(DataPath.S3_DAILY_WATCH_ENT_PLATFORM, country)
            % get_dates_str_backwards(date, cum_days)
        )
        .groupBy("dw_p_id", "content_id")
        .agg(F.sum("watch_time").alias("cum_watch_time"))
        .withColumn(
            "cum_watch_time", (F.col("cum_watch_time") / 3600).cast(FloatType())
        )
    )
    allwt_label = (
        spark.read.parquet(
            data_path(DataPath.S3_DAILY_WATCH_ENT_PLATFORM, country)
            % get_dates_str_backwards(date, cum_days)
        )
        .groupBy("dw_p_id")
        .agg(F.sum("watch_time").alias("all_watch_time"))
        .withColumn(
            "all_watch_time", (F.col("all_watch_time") / 3600).cast(FloatType())
        )
    )
    return label, allwt_label


def generate_features(
    spark: SQLContext,
    discovery_date,
    tenant,
    country,
    variant,
    discovery_basepath,
):
    feature_path = data_path(
        TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_FEATURE, country
    ) % (
        variant,
        discovery_date,
    )
    if is_s3_path_success(feature_path):
        print("load cached feature", feature_path)
        return spark.read.parquet(feature_path)

    print("generate features for ", discovery_date)
    metadata = load_metadata(spark, discovery_date, tenant)

    bc_metadata = spark.sparkSession.sparkContext.broadcast(metadata)

    def process_row_fd(row):
        example_dict = extract_feature_using_fd(
            metadata=bc_metadata.value,
            dw_p_id=row["dw_p_id"],
            content_id=row["content_id"],
            langauge_id=row["language"],
            timestamp=row["timestamp"],
            events=row["events"],
            past_wt_offline=row["past_wt"],
            country=country,
        )
        if example_dict is not None:
            # dummy column to comply with schema
            example_dict["day"] = -1
            example_dict["cum_watch_time"] = -1.0
            example_dict["all_watch_time"] = -1.0
            return dict_to_spark_row(CumWatchDataSchema, example_dict)
        else:
            return None

    print("write features to", feature_path)

    base = spark.read.parquet(
        data_path(discovery_basepath, country) % discovery_date
    ).filter(F.col("watch_time") > 600)

    events = load_tpfy_predict_events(spark, discovery_date, country)
    if "past_wt" not in base.columns:
        base = base.withColumn("past_wt", F.lit(0))
    base.select("dw_p_id", "content_id", "language", "timestamp", "past_wt").join(
        events, on="dw_p_id", how="inner"
    ).rdd.map(process_row_fd).filter(lambda row: row is not None).toDF(
        CumWatchDataSchema.as_spark_schema()
    ).repartition(
        32
    ).write.mode(
        "overwrite"
    ).parquet(
        feature_path
    )

    bc_metadata.unpersist()
    print("done feature", discovery_date)

    return spark.read.parquet(feature_path)


def get_example_dataset(spark, country, date, cum_days, variant, discovery_basepath):
    discovery_date = get_diff_date(date, -cum_days + 1)

    discovery_dataset_path = data_path(discovery_basepath, country) % discovery_date
    print("run cum example from %s to %s" % (discovery_date, date))
    if not is_s3_path_success(discovery_dataset_path):
        print("discovery dataset on %s not exist; skip" % date)
        return None
    discovery_dataset = (
        spark.read.parquet(discovery_dataset_path)
        .filter(F.col("watch_time") > 600)
        .select("dw_p_id", "content_id")
    )

    feature_dataset_path = data_path(
        TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_FEATURE, country
    ) % (
        variant,
        discovery_date,
    )
    feature_dataset = spark.read.parquet(feature_dataset_path).drop(
        "day", "cum_watch_time", "all_watch_time"
    )

    label, allwt_label = get_labels(spark, date, country, cum_days)

    example_dataset = (
        discovery_dataset.join(feature_dataset, on=["dw_p_id", "content_id"])
        .join(label, on=["dw_p_id", "content_id"])
        .join(allwt_label, on=["dw_p_id"])
        .withColumn("day", F.lit(cum_days - 1))
        .withColumn("country", F.lit(country))
    )
    return example_dataset


def run_for_country(
    spark,
    country,
    date,
    variant,
    watched_sample_rate,
    agg_prog,
    gen_example=True,
):
    if variant is None:
        variant = ""
    elif len(variant) > 0 and not variant.startswith("-"):
        variant = "-" + variant

    # generate new watches today for future use
    if watched_sample_rate > 0:
        discovery_basepath = TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_BASE_WATCHED
        if variant:
            variant = "-watched" + variant
        else:
            variant = "-watched"
    else:
        discovery_basepath = TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_BASE
    print("variant", variant)
    print("watched sample rate", watched_sample_rate)
    print("discovery base", discovery_basepath)

    run_discovery(
        spark,
        date,
        country,
        watched_sample_rate=watched_sample_rate,
        discovery_basepath=discovery_basepath,
    )

    generate_features(
        spark,
        date,
        tenant=TENANT,
        country=country,
        variant=variant,
        discovery_basepath=discovery_basepath,
    )

    if not gen_example:
        return

    MAX_CUM_DAYS = 7
    example_dataset = get_example_dataset(
        spark,
        country,
        date,
        MAX_CUM_DAYS,
        variant=variant,
        discovery_basepath=discovery_basepath,
    )
    if example_dataset is not None:
        dest_path = data_path(TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_EXAMPLE_D7) % (
            variant,
            date,
        )
        print("write full dataset to", dest_path)
        example_dataset.repartition(32, "dw_p_id").write.mode("overwrite").parquet(
            dest_path
        )

    if agg_prog:
        prog_dest_path = data_path(
            TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_EXAMPLE_PROG
        ) % (
            variant,
            date,
        )
        print("write prog dataset to", prog_dest_path)
        remove_s3_folder(prog_dest_path)
        for cum_day in range(1, MAX_CUM_DAYS):
            part = get_example_dataset(
                spark,
                country,
                date,
                cum_day,
                variant=variant,
                discovery_basepath=discovery_basepath,
            )
            if part is None:
                continue
            part.repartition(32, "dw_p_id").write.mode("append").parquet(prog_dest_path)

        print("done")


def run(args):
    end_date = args.end_date
    days = args.days
    spark_context = create_spark_context(
        "TPFY cum watch dataset %s" % end_date, enable_hive=True
    )

    sql_context = create_sql_context(spark_context)
    set_partitions(sql_context, 1024, adjust=True)

    variant = args.variant

    for country in tenant_countries(args.countries):
        for date in get_dates_list_backwards(end_date, days)[::-1]:
            agg_prog = date >= get_diff_date(end_date, -2)
            print("start running for country:", country, date)
            run_for_country(
                sql_context,
                country,
                date,
                variant=variant,
                watched_sample_rate=0,
                agg_prog=agg_prog,
                gen_example=not args.no_gen_example,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("end_date", type=str, help="end date, YYYY-mm-dd")
    parser.add_argument("days", type=int, default=1)
    parser.add_argument("--use_cms3", action="store_true", help="deprecated; ignore")
    parser.add_argument("--use_fd", action="store_true")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--no_gen_example", action="store_true")
    parser.add_argument(
        "--countries",
        type=str,
        help="countries to run, separated with comma. "
        "default is None. fallback to region countries",
    )
    args = parser.parse_args()
    run(args)
