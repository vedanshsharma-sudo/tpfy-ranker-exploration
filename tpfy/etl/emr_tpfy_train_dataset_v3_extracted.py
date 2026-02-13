import argparse
import sys
import time
from typing import Dict

import cityhash
import pyspark.sql.functions as F
import s3fs
from pyspark.sql.types import FloatType, BooleanType

from common.config import TENANT
from common.config.utils import data_path, tenant_countries
from common.s3_utils import *
from common.slack_utils import exception_to_slack
from common.spark_utils import (
    create_spark_context,
    create_sql_context,
    set_partitions,
)
from common.time_utils import get_dates_list_backwards, get_dates_str_backwards
from tpfy.common import TpfyDataPath
from tpfy.etl.emr_train_dataset import (
    load_raw_complete_ent_watches,
    generate_daily_impr_examples,
)
from tpfy.etl.feature_extractor_spark import (
    emr_load_metadata,
    extract_features,
    mtl_extract_features,
    ExtractConfig,
)
from tpfy.etl.lib import load_tpfy_predict_events
from tpfy.etl.schema import TpfyDatasetSchema, TpfyMtlDatasetSchema


class TenantTpfyEtlConfig:
    def __init__(
        self,
        daily_lw_sample_rate,
        agg_lw_sample_rate,
        num_random_neg,
        daily_mtl_ns_rate=0.1,
        daily_mtl_lw_sample_rate=1.0,
        mtl_num_random_neg=7,
    ):
        self.daily_lw_sample_rate = daily_lw_sample_rate
        self.agg_lw_sample_rate = agg_lw_sample_rate
        self.num_random_neg = num_random_neg
        self.daily_mtl_ns_rate = daily_mtl_ns_rate
        self.daily_mtl_lw_sample_rate = daily_mtl_lw_sample_rate
        self.mtl_num_random_neg = mtl_num_random_neg

    def __str__(self):
        return f"daily_lw_sample_rate: {self.daily_lw_sample_rate}, agg_lw_sample_rate: {self.agg_lw_sample_rate}, random_neg: {self.num_random_neg}"


tpfy_etl_config_dict: Dict[str, TenantTpfyEtlConfig] = {
    "in": TenantTpfyEtlConfig(
        daily_lw_sample_rate=0.1,
        agg_lw_sample_rate=0.4,
        num_random_neg=15,
        daily_mtl_ns_rate=0.08,
        daily_mtl_lw_sample_rate=0.1,
        mtl_num_random_neg=7,
    ),
    "SEA": TenantTpfyEtlConfig(
        daily_lw_sample_rate=0.1,
        agg_lw_sample_rate=0.4,
        num_random_neg=15,
        daily_mtl_lw_sample_rate=0.1,
    ),
    "MEA": TenantTpfyEtlConfig(
        daily_lw_sample_rate=1.0,
        agg_lw_sample_rate=0.3,
        num_random_neg=4,
        daily_mtl_lw_sample_rate=0.1,
    ),
}


def normalize_variant_suffix(variant):
    if variant and not variant.startswith("-"):
        variant = "-" + variant
    return variant


def extract_daily_examples_features(
    sql_context,
    bc_metadata,
    country,
    date,
    etl_config: TenantTpfyEtlConfig,
    variant,
    extract_config,
    use_cached_raw=True,
):
    if variant and not variant.startswith("-"):
        variant = "-" + variant
    daily_example_path = (
        data_path(TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_IMPR_EXAMPLES, country) % date
    )
    if not is_s3_path_success(daily_example_path):
        raise Exception(f"daily example not ready {daily_example_path}")

    raw_cache_path = data_path(TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_RAW, country) % date
    if is_s3_file_exist(os.path.join(raw_cache_path, "_SUCCESS")) and use_cached_raw:
        print("use cached raw")
    else:
        print("generate raw data cache")
        raw_complete_watches = load_raw_complete_ent_watches(
            sql_context, date, dw_pid_filter=lambda _: F.lit(True), country=country
        )
        examples = sql_context.read.parquet(daily_example_path)
        events = load_tpfy_predict_events(sql_context, date, country)

        raw_data = examples.join(events, on="dw_p_id", how="inner").join(
            raw_complete_watches, on="dw_p_id", how="left"
        )
        raw_data.repartition(256, ["dw_p_id"]).write.mode("overwrite").parquet(
            raw_cache_path
        )
        print("done")

    metadata = bc_metadata.value

    candidate_ids = list(metadata.movies[country].keys()) + list(
        metadata.tv_shows[country].keys()
    )
    bc_candidate_ids = sql_context.sparkSession.sparkContext.broadcast(candidate_ids)
    raw_data = sql_context.read.parquet(raw_cache_path)

    dest_path_template = TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_EXTRACTED_EXAMPLES

    extract_dest_path = data_path(dest_path_template, country) % (variant, date)
    print("extracted goes to ", extract_dest_path)
    remove_s3_folder(extract_dest_path)

    start_time = time.time()
    raw_data.select(
        "dw_p_id",
        "tpfy_examples",
        "lastwatch_examples",
        "events",
        "complete_watches",
    ).rdd.flatMap(
        lambda row: extract_features(
            dw_p_id=row["dw_p_id"],
            tpfy_examples=row["tpfy_examples"],
            lw_examples=row["lastwatch_examples"],
            events=row["events"],
            raw_complete_watches=row["complete_watches"],
            metadata=bc_metadata.value,
            candidate_ids=bc_candidate_ids.value,
            country=country,
            lw_sample_rate=etl_config.daily_lw_sample_rate,
            num_random_neg=etl_config.num_random_neg,
            extract_config=extract_config,
        )
    ).toDF(
        TpfyDatasetSchema.as_spark_schema()
    ).repartition(
        32, ["dw_p_id"]
    ).write.mode(
        "overwrite"
    ).parquet(
        extract_dest_path
    )
    extract_used = time.time() - start_time
    print(f"done; used {extract_used} seconds")

    result_df = sql_context.read.parquet(extract_dest_path)
    result_df.groupBy("flag").agg(
        F.sum(F.col("label").getItem(0)).alias("num_positive"),
        F.count(F.lit(1)).alias("total"),
    ).show()

    if True:
        start_time = time.time()
        mtl_extract_dest_path = data_path(
            TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_MTL_EXTRACTED_EXAMPLES, country
        ) % (variant, date)
        print("extracted mtl example features goes to ", mtl_extract_dest_path)
        remove_s3_folder(mtl_extract_dest_path)

        raw_data.select(
            "dw_p_id",
            "tpfy_mtl_examples",
            "lastwatch_examples",
            "events",
            "complete_watches",
        ).rdd.flatMap(
            lambda row: mtl_extract_features(
                dw_p_id=row["dw_p_id"],
                tpfy_mtl_examples=row["tpfy_mtl_examples"],
                lw_examples=row["lastwatch_examples"],
                events=row["events"],
                raw_complete_watches=row["complete_watches"],
                metadata=bc_metadata.value,
                candidate_ids=bc_candidate_ids.value,
                country=country,
                lw_sample_rate=etl_config.daily_mtl_lw_sample_rate,
                num_random_neg=etl_config.mtl_num_random_neg,
                extract_config=extract_config,
            )
        ).toDF(
            TpfyMtlDatasetSchema.as_spark_schema()
        ).repartition(
            32, ["dw_p_id"]
        ).write.mode(
            "overwrite"
        ).parquet(
            mtl_extract_dest_path
        )
        extract_used = time.time() - start_time
        print(f"done; used {extract_used} seconds")

    bc_candidate_ids.unpersist()


def merge_tenant_daily_dataset(sql_context, tenant, countries, date, variant):
    if len(countries) == 1 and countries[0] == tenant:
        print(f"skip merging tenant {tenant} daily dataset")
        return
    variant = normalize_variant_suffix(variant)

    src_paths = [
        data_path(TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_EXTRACTED_EXAMPLES, country)
        % (variant, date)
        for country in countries
    ]

    dst_path = data_path(
        TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_EXTRACTED_EXAMPLES, tenant
    ) % (variant, date)
    remove_s3_folder(dst_path)

    print("merge daily dataset source src", src_paths)
    print("merge daily dataset source dst", dst_path)

    sql_context.read.parquet(*src_paths).repartition(32, ["dw_p_id"]).write.mode(
        "overwrite"
    ).parquet(dst_path)
    print("done")


def merge_mtl_tenant_daily_dataset(sql_context, tenant, countries, date, variant):
    if len(countries) == 1 and countries[0] == tenant:
        print(f"skip merging tenant {tenant} daily dataset")
        return
    variant = normalize_variant_suffix(variant)

    src_paths = [
        data_path(TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_MTL_EXTRACTED_EXAMPLES, country)
        % (variant, date)
        for country in countries
    ]

    dst_path = data_path(
        TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_MTL_EXTRACTED_EXAMPLES, tenant
    ) % (variant, date)
    remove_s3_folder(dst_path)

    print("merge mtl daily dataset source src", src_paths)
    print("merge mtl daily dataset source dst", dst_path)

    sql_context.read.parquet(*src_paths).repartition(32, ["dw_p_id"]).write.mode(
        "overwrite"
    ).parquet(dst_path)
    print("done")


@F.udf(returnType=FloatType())
def sample_value(dw_p_id, cd, end_dt):
    seed = cityhash.CityHash32(dw_p_id + str(cd) + end_dt)
    rng = random.Random(seed)
    return rng.random()


def run_agg_data(
    sql_context,
    tenant,
    countries,
    end_date,
    days,
    random_sample_rate,
    variant,
):
    if variant and not variant.startswith("-"):
        variant = "-" + variant
    print(f"agg training data in tenant {tenant}; countries {countries}; days ${days}")

    src_template = TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_EXTRACTED_EXAMPLES
    dst_template = TpfyDataPath.S3_TPFY_IMPR_V3_AGG_EXTRACTED_EXAMPLES_VAR

    source_df = None
    for dt in get_dates_list_backwards(end_date, days):
        path = data_path(src_template, tenant) % (variant, dt)
        if not is_s3_path_success(path):
            raise Exception(f"{path} not successful")
    src_path = data_path(src_template, tenant) % (
        variant,
        get_dates_str_backwards(end_date, days),
    )
    base_path = src_path[: src_path.index("cd=")]
    print(f"agg source {tenant}: {src_path}")
    country_df = sql_context.read.option("basePath", base_path).parquet(src_path)

    if source_df is None:
        source_df = country_df
    else:
        source_df = source_df.unionByName(country_df)

    dst_path = data_path(dst_template, tenant) % (variant, end_date)
    print(f"agg dst {dst_path}")
    remove_s3_folder(dst_path)
    df = source_df.filter(
        (F.col("flag") == 0)
        | (
            (F.col("flag") == 1)
            & (sample_value("dw_p_id", "cd", F.lit(end_date)) < random_sample_rate)
        )
    )
    df.repartition(256, ["dw_p_id"]).write.mode("overwrite").parquet(dst_path)

    stats_df = (
        df.groupBy(F.col("task").getItem(0).alias("task"))
        .agg(F.count(F.lit(1)).alias("count"))
        .collect()
    )
    total_count = df.count()

    task_weights = {}
    task_counts = {}
    for row in stats_df:
        task = row["task"]
        count = row["count"]
        weight = count / total_count
        task_weights[str(task)] = weight
        task_counts[str(task)] = count

    stats = {"task_weights": task_weights, "task_counts": task_counts}
    print("stats", stats)
    fs = s3fs.S3FileSystem(use_ssl=False)
    with fs.open(os.path.join(dst_path, "stats.json"), "w") as f:
        json.dump(stats, f)
    set_s3_file_acl(os.path.join(dst_path, "stats.json"))


def run_agg_mtl_data(
    sql_context,
    tenant,
    countries,
    end_date,
    days,
    variant,
):
    if variant and not variant.startswith("-"):
        variant = "-" + variant
    print(f"agg training data in tenant {tenant}; countries {countries}; days ${days}")

    src_template = TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_MTL_EXTRACTED_EXAMPLES
    dst_template = TpfyDataPath.S3_TPFY_IMPR_V3_AGG_MTL_EXTRACTED_EXAMPLES_VAR

    for dt in get_dates_list_backwards(end_date, days):
        path = data_path(src_template, tenant) % (variant, dt)
        if not is_s3_path_success(path):
            raise Exception(f"{path} not successful")
    src_path = data_path(src_template, tenant) % (
        variant,
        get_dates_str_backwards(end_date, days),
    )
    base_path = src_path[: src_path.index("cd=")]
    print(f"agg source {tenant}: {src_path}")
    source_df = sql_context.read.option("basePath", base_path).parquet(src_path)

    dst_path = data_path(dst_template, tenant) % (variant, end_date)
    print(f"agg dst {dst_path}")
    remove_s3_folder(dst_path)
    df = source_df
    df.repartition(256, ["dw_p_id"]).write.mode("overwrite").parquet(dst_path)

    # randomwatch: task==1
    # click: task == 0
    # postclick: task == 0, click == 1
    stats_df = (
        df.groupBy(F.col("task").getItem(0).alias("task"))
        .agg(F.count(F.lit(1)).alias("count"))
        .collect()
    )
    postclick_count = df.filter(
        (F.col("task").getItem(0) == 0) & (F.col("click").getItem(0) > 0)
    ).count()
    total_count = df.count()

    task_weights = {}
    task_counts = {}
    for row in stats_df:
        task = row["task"]
        count = row["count"]
        weight = count / total_count
        task_name = {0: "click", 1: "random_watch"}[task]
        task_weights[task_name] = weight
        task_counts[task_name] = count

    task_weights["postclick"] = postclick_count / total_count
    task_counts["postclick"] = postclick_count

    stats = {"task_weights": task_weights, "task_counts": task_counts}
    print("stats", stats)
    fs = s3fs.S3FileSystem(use_ssl=False)
    with fs.open(os.path.join(dst_path, "stats.json"), "w") as f:
        json.dump(stats, f)
    set_s3_file_acl(os.path.join(dst_path, "stats.json"))


@exception_to_slack(title="tpfy-exdeepfm_training_dataset_extracted")
def run(args):
    print("python version", sys.version)

    end_date = args.date
    days = args.days
    print("end date: ", end_date, ", days: ", days)

    spark_context = create_spark_context("TPFY ctx dataset", enable_hive=True)

    sql_context = create_sql_context(spark_context)
    set_partitions(sql_context, 2048, adjust=True)

    countries_no_override = tenant_countries()
    countries = tenant_countries(args.countries)

    etl_config = tpfy_etl_config_dict.get(TENANT, None)
    if etl_config is None:
        raise Exception(f"etl config for {TENANT} missing")
    if args.num_random_neg is not None:
        etl_config.num_random_neg = args.num_random_neg
    if args.lw_sample_rate is not None and args.lw_sample_rate > 0:
        if args.agg:
            etl_config.agg_lw_sample_rate = args.lw_sample_rate
        else:
            etl_config.daily_lw_sample_rate = args.lw_sample_rate

    if args.agg:
        print("agg etl config", etl_config)

        run_agg_data(
            sql_context,
            TENANT,
            countries,
            end_date,
            days,
            etl_config.agg_lw_sample_rate,
            args.variant,
        )
        if True:
            run_agg_mtl_data(
                sql_context,
                TENANT,
                countries,
                end_date,
                days=7,
                variant=args.variant,
            )
    else:
        print("daily etl config", etl_config)
        dw_pid_filter = lambda _: F.lit(True)
        for date in get_dates_list_backwards(end_date, args.days):
            metadata = emr_load_metadata(
                sql_context,
                date,
                TENANT,
                countries_no_override,
                use_cached_metadata=args.use_cached_metadata,
            )

            if not (
                hasattr(metadata, "discover_popularity")
                and hasattr(metadata, "state_nolang_popularity")
            ):
                metadata.load_popularity()

            for country in countries:
                print(f"generate daily examples for country:{country}, date {date}")
                convert_to_show_content_id_map = {}
                for cid, episode in metadata.episode_repo.get_items(country):
                    show_content_id = episode.show_content_id
                    if (
                        show_content_id
                        and show_content_id in metadata.tv_shows[country]
                    ):
                        convert_to_show_content_id_map[cid] = show_content_id

                print(
                    "len convert_to_show_content_id_map",
                    len(convert_to_show_content_id_map),
                )
                bc_convert_to_show_content_id_map = (
                    sql_context.sparkSession.sparkContext.broadcast(
                        convert_to_show_content_id_map
                    )
                )
                generate_daily_impr_examples(
                    sql_context,
                    date,
                    dw_pid_filter,
                    country,
                    bc_convert_to_show_content_id_map,
                    mtl_uni_ns_rate=etl_config.daily_mtl_ns_rate,
                )
                bc_convert_to_show_content_id_map.unpersist()
                print("generate daily examples for country done")

            bc_metadata = sql_context.sparkSession.sparkContext.broadcast(metadata)
            for country in countries:
                print(f"extract daily features for country:{country}, date {date}")
                extract_config = ExtractConfig()
                extract_daily_examples_features(
                    sql_context,
                    bc_metadata,
                    country,
                    date,
                    etl_config=etl_config,
                    variant=args.variant,
                    extract_config=extract_config,
                    use_cached_raw=args.use_cached_raw,
                )
            bc_metadata.unpersist()

            merge_tenant_daily_dataset(
                sql_context, TENANT, countries, date, variant=args.variant
            )
            merge_mtl_tenant_daily_dataset(
                sql_context, TENANT, countries, date, variant=args.variant
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("date", type=str, help="end date, YYYY-mm-dd")
    parser.add_argument("days", type=int)
    parser.add_argument("unused_subsampling", nargs="?", type=float, default=1.0)
    parser.add_argument("--lw_sample_rate", type=float, default=None)
    parser.add_argument("--num_random_neg", type=int, default=None)
    parser.add_argument("--use_cms3", action="store_true", help="deprecated; ignore")
    parser.add_argument("--fid_map23", action="store_true", help="deprecated; ignore")
    parser.add_argument("--fid_map32", action="store_true", help="deprecated; ignore")
    parser.add_argument("--use_cached_raw", default=True, action="store_true")
    parser.add_argument(
        "--no-use_cached_raw", dest="use_cached_raw", action="store_false"
    )
    parser.add_argument("--use_cached_metadata", default=True, action="store_true")
    parser.add_argument(
        "--no-use_cached_metadata", dest="use_cached_metadata", action="store_false"
    )
    parser.add_argument("--agg", action="store_true")
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument(
        "--countries",
        type=str,
        help="countries to run, separated with comma. "
        "default is None. fallback to region countries",
    )
    args = parser.parse_args()
    run(args)
