from collections import namedtuple, defaultdict
from itertools import chain

import pyspark.sql.functions as F
from pyspark.sql.functions import (
    col,
)
from pyspark.sql.types import *
from pyspark.sql.types import Row

from common.config.constants import DataPath, EMPTY_USER_DW_PID
from common.config.utils import data_path, get_config
from common.data.behavior.watch import Watch2
from common.s3_utils import *
from common.time_utils import timestamp, get_diff_date
from tpfy.common import TpfyDataPath


def load_raw_complete_watches(sql_context, date, country):
    return (
        sql_context.read.option("mergeSchema", "false")
        .parquet(data_path(DataPath.S3_GROUP_WATCH_UBS, country) % date)
        .filter(col("dw_p_id").isNotNull() & (col("dw_p_id") != EMPTY_USER_DW_PID))
    )


def load_raw_complete_ent_watches(sql_context, date, dw_pid_filter, country):
    df = load_raw_complete_watches(sql_context, date, country)
    return df.filter(dw_pid_filter("dw_p_id")).select(
        F.col("dw_p_id"), F.col("watches").alias("complete_watches")
    )


def load_daily_ent_watch(sql_context, date, dw_pid_filter, country):
    df = (
        sql_context.read.option("mergeSchema", "false")
        .parquet(data_path(DataPath.S3_DAILY_WATCH_ENT, country) % date)
        .filter(dw_pid_filter("dw_p_id"))
        .select(
            "dw_p_id",
            F.struct(
                "content_id",
                "language",
                "watch_time",
                "first_watch_ts",
                "last_watch_ts",
            ).alias("watch"),
        )
        .groupBy("dw_p_id")
        .agg(F.collect_list("watch").alias("watches"))
    )

    return df


def load_daily_add_watchlist(sql_context, date, dw_pid_filter, country):
    WATCHLIST_ADD_ACTION = 1
    df = (
        sql_context.read.option("mergeSchema", "false")
        .parquet(data_path(DataPath.S3_DAILY_WATCHLIST, country) % date)
        .filter(dw_pid_filter("dw_p_id"))
        .filter(F.col("action") == WATCHLIST_ADD_ACTION)
        .select(
            "dw_p_id",
            F.struct(
                "content_id",  # long
                "timestamp",
            ).alias("item"),
        )
        .groupBy("dw_p_id")
        .agg(F.collect_list("item").alias("watchlist_adds"))
    )
    return df


def load_daily_paywall_view(sql_context, date, dw_pid_filter, country):
    df = (
        sql_context.read.parquet(
            data_path(DataPath.S3_DAILY_PAYWALL_VIEW, country) % date
        )
        .filter(dw_pid_filter("dw_p_id"))
        .filter(col("normalized_content_id").isNotNull())
        .select(
            "dw_p_id",
            col("normalized_content_id").alias("content_id"),  # string
            col("received_at").alias("timestamp"),
        )
    )
    df = df.groupBy("dw_p_id").agg(
        F.collect_list(F.struct(["content_id", "timestamp"])).alias("paywall_views")
    )
    return df


def load_daily_tray_impression(spark, date, dw_pid_filter, country):
    is_tpfy_tray = F.col("tray_id") == "p13n-tpfy"
    # is_masthead_tray = (F.col("tray_name") == "Spotlight")
    is_reco_collection_tray = F.col("full_tray_id").startswith("reco-tpfy")
    is_tpv2_tray = F.col("full_tray_id").startswith("reco-tpv2")

    tray_filter = is_tpfy_tray | is_reco_collection_tray | is_tpv2_tray

    s3_path = data_path(DataPath.S3_TRAY_IMPRESSION, country) % date

    sentinel_path = os.path.join(s3_path, "_SUCCESS")
    if not is_s3_file_exist(sentinel_path):
        raise Exception(f"tray impression in {sentinel_path} is unavailable; fail")
    return (
        spark.read.parquet(s3_path)
        .filter(dw_pid_filter("dw_p_id"))
        .filter(tray_filter)
        .select(
            "dw_p_id",
            F.struct("full_tray_id", "timestamp", "content_id", "tile_position").alias(
                "impr"
            ),
        )
        .groupBy("dw_p_id")
        .agg(F.collect_list("impr").alias("tpfy_impressions"))
    )


def load_daily_tray_ent_watch(spark, date, dw_pid_filter, country):
    s3_path = data_path(DataPath.S3_DAILY_TRAY_ENT_WATCH_PLATFORM, country) % date
    sentinel_path = os.path.join(s3_path, "_SUCCESS")
    if not is_s3_file_exist(sentinel_path):
        raise Exception(f"daily tray ent watch {sentinel_path} not ready")

    return spark.read.parquet(s3_path).filter(dw_pid_filter("dw_p_id"))


def load_daily_tray_tile_click(spark, date, dw_pid_filter, country):
    s3_path = data_path(DataPath.S3_DAILY_CLICK, country) % date
    sentinel_path = os.path.join(s3_path, "_SUCCESS")
    if not is_s3_file_exist(sentinel_path):
        raise Exception(f"tile click {sentinel_path} not ready")

    VALID_DWPID_COND = (
        dw_pid_filter("dw_p_id")
        & col("dw_p_id").isNotNull()
        & (col("dw_p_id") != "")
        & (col("dw_p_id") != EMPTY_USER_DW_PID)
    )

    df = (
        spark.read.parquet(s3_path)
        .filter(VALID_DWPID_COND)
        .withColumn("full_tray_id", col("tray_id"))
        .select(
            "dw_p_id",
            F.struct("content_id", "timestamp", "full_tray_id").alias("tile_click"),
        )
        .groupBy("dw_p_id")
        .agg(F.collect_list("tile_click").alias("tile_clicks"))
    )
    return df


POSITIVE_WATCH_THRESHOLD = 600

ImprExample = namedtuple(
    "ImprSample",
    ["tray_id", "content_id", "timestamp", "label", "watch_time", "is_inter_tray"],
)
ImprExampleRowType = StructType(
    [
        StructField("tray_id", StringType()),
        StructField("content_id", LongType()),
        StructField("timestamp", LongType()),
        StructField("label", LongType()),
        StructField("watch_time", LongType()),
        StructField("is_inter_tray", BooleanType()),
    ]
)

MtlExample = namedtuple(
    "MtlExample",
    [
        "tray_id",
        "content_id",
        "timestamp",
        "click",
        "watch",
        "watch_time",
        "paywall_view",
        "add_watchlist",
    ],
)
MtlExampleRowType = StructType(
    [
        StructField("tray_id", StringType()),
        StructField("content_id", LongType()),
        StructField("timestamp", LongType()),
        StructField("click", IntegerType()),
        StructField("watch", IntegerType()),
        StructField("watch_time", IntegerType()),
        StructField("paywall_view", IntegerType()),
        StructField("add_watchlist", IntegerType()),
    ]
)


class TrayAggregation:
    def __init__(self):
        # content_id to watch_time
        self.watches = {}
        # content_id to timestamp
        self.impressions = {}

    def add_impression(self, content_id: int, timestamp):
        if content_id in self.impressions:
            ts = self.impressions[content_id]
            self.impressions[content_id] = min(ts, timestamp)
        else:
            self.impressions[content_id] = timestamp

    def add_watch(self, content_id: int, watch_time):
        if content_id not in self.watches:
            self.watches[content_id] = watch_time
        else:
            self.watches[content_id] += watch_time


def safe_convert_int(content_id):
    try:
        return int(content_id)
    except:
        return None


def generate_user_samples_v3(tray_impressions, tray_watches, raw_complete_watches):
    if not tray_watches:
        return []
    if not tray_impressions:
        return []

    complete_watches = Watch2.parse_watches_fn()(raw_complete_watches)
    complete_watched_content_ids = {Watch2(*w).content_id for w in complete_watches}

    # TODO: group by platform and page
    tray_id_to_agg = {}
    for tray_impr in tray_impressions:
        tray_id = tray_impr.full_tray_id
        ts = tray_impr.timestamp
        tray_agg = tray_id_to_agg.get(tray_id, None)
        if tray_agg is None:
            tray_agg = TrayAggregation()
            tray_id_to_agg[tray_id] = tray_agg
        content_id = safe_convert_int(tray_impr["content_id"])
        if not content_id:
            continue
        tray_agg.add_impression(content_id, ts)

    for w in tray_watches:
        tray_id = w.tray_id
        content_id = w.content_id
        if tray_id not in tray_id_to_agg:
            continue
        tray_id_to_agg[tray_id].add_watch(content_id, w.watch_time)

    examples = []
    total_positives = 0
    neg_sampling_multi = 5
    pending_tray_neg_examples = []
    for tray_id, agg in tray_id_to_agg.items():
        tray_id_hash = hash(tray_id)

        tray_positives = []
        tray_negatives = []

        if len(agg.watches) == 0:
            continue
        for content_id, first_ts in agg.impressions.items():
            if content_id in agg.watches:
                watch_time = agg.watches[content_id]
                if watch_time > POSITIVE_WATCH_THRESHOLD:
                    tray_positives.append(
                        ImprExample(
                            tray_id=tray_id,
                            content_id=content_id,
                            timestamp=first_ts,
                            label=1,
                            watch_time=int(watch_time),
                            is_inter_tray=False,
                        )
                    )
            elif content_id in complete_watched_content_ids:
                continue
            else:
                tray_negatives.append(
                    ImprExample(
                        tray_id=tray_id,
                        content_id=content_id,
                        timestamp=first_ts,
                        label=0,
                        watch_time=0,
                        is_inter_tray=False,
                    )
                )
        if len(tray_positives) == 0:
            pending_tray_neg_examples.extend(tray_negatives)
            continue
        if len(tray_negatives) > len(tray_positives) * neg_sampling_multi:
            random.shuffle(tray_negatives)
            tray_negatives = tray_negatives[: len(tray_positives) * neg_sampling_multi]
        examples.extend(tray_positives)
        examples.extend(tray_negatives)
        total_positives += len(tray_positives)

    if total_positives > 0 and len(pending_tray_neg_examples) > 0:
        # inter tray negatives
        num = min(total_positives, len(pending_tray_neg_examples))
        inter_tray_negatives = [
            ex._replace(is_inter_tray=True)
            for ex in random.sample(pending_tray_neg_examples, num)
        ]
        examples.extend(inter_tray_negatives)

    rows = [Row(**ex._asdict()) for ex in examples]
    return rows


def generate_lastwatch_examples(ent_watches, raw_past_complete_watches):
    past_complete_watches = Watch2.parse_watches_fn()(raw_past_complete_watches)
    past_watched_content_ids = {w.content_id for w in past_complete_watches}

    if ent_watches is None or len(ent_watches) == 0:
        return []
    for w in sorted(ent_watches, key=lambda w: w.first_watch_ts):  # first valid watch
        assert isinstance(w.content_id, int)
        if (
            w.watch_time > POSITIVE_WATCH_THRESHOLD
            and w.content_id not in past_watched_content_ids
        ):
            ex = ImprExample(
                tray_id="",
                content_id=w.content_id,
                timestamp=w.first_watch_ts,
                label=1,
                watch_time=int(w.watch_time),
                is_inter_tray=False,
            )
            return [Row(**ex._asdict())]
    return []


def is_reco_tray(tray_id):
    return tray_id and (
        tray_id.startswith("reco-tpfy")
        or tray_id.startswith("p13n-tpfy")
        or tray_id.startswith("reco-tpv2")
    )


def generate_mtl_examples_v3_builder(bc_convert_to_show_content_id_map, uni_ns_rate):
    def generate_mtl_examples_v3(
        tray_impressions,
        ent_watches,
        raw_complete_watches,
        tile_clicks,
        paywall_views,
        watchlist_adds,
    ):
        if not tray_impressions:
            return []
        if not tile_clicks:
            return []

        # complete_watches = Watch2.parse_watches_fn()(raw_complete_watches)

        convert_to_show_content_id_map = (
            bc_convert_to_show_content_id_map.value
        )  # episodes cid to show cid; int to int
        paywall_view_final_ts = {}
        if paywall_views:
            for view in sorted(paywall_views, key=lambda view: view["timestamp"]):
                content_id = safe_convert_int(view["content_id"])
                ts = int(view["timestamp"].timestamp())  # datetime to timestamp
                if not content_id:
                    continue
                if content_id in convert_to_show_content_id_map:
                    content_id = convert_to_show_content_id_map[content_id]

                paywall_view_final_ts[content_id] = ts

        watchlist_add_final_ts = {}
        if watchlist_adds:
            for wl_add in sorted(watchlist_adds, key=lambda wl: wl["timestamp"]):
                content_id = wl_add["content_id"]  # int
                assert isinstance(content_id, int)
                ts = wl_add["timestamp"]
                watchlist_add_final_ts[content_id] = ts

        tray_id_to_agg = {}
        for tray_impr in tray_impressions:
            tray_id = tray_impr.full_tray_id
            if not is_reco_tray(tray_id):
                continue
            ts = tray_impr.timestamp
            tray_agg = tray_id_to_agg.get(tray_id, None)
            if tray_agg is None:
                tray_agg = TrayAggregation()
                tray_id_to_agg[tray_id] = tray_agg

            content_id = safe_convert_int(tray_impr["content_id"])
            if not content_id:
                continue
            tray_agg.add_impression(content_id, ts)

        today_content_watch_time = defaultdict(int)
        if ent_watches:
            for w in ent_watches:
                content_id = w["content_id"]
                watch_time = w["watch_time"]
                assert isinstance(content_id, int)
                if watch_time > 0:
                    today_content_watch_time[content_id] += w["watch_time"]

        tile_clicks = sorted(tile_clicks, key=lambda click: click["timestamp"])
        clicked_content_ids = set()
        for click in tile_clicks:
            clicked_content_ids.add(click["content_id"])  # all clicks (besides tpfy)

        positives = []
        negatives = []
        visited_content_ids = set()
        for click in tile_clicks:
            tray_id = click["full_tray_id"]
            if not is_reco_tray(tray_id):
                continue

            content_id = click["content_id"]  # int
            timestamp = click["timestamp"]
            assert isinstance(content_id, int)

            if content_id in visited_content_ids:
                continue

            tray_agg = tray_id_to_agg.get(tray_id)
            if tray_agg is None:
                continue
            first_impr_ts = tray_agg.impressions.get(content_id, 0)
            if first_impr_ts == 0:
                continue

            watch_time = int(today_content_watch_time.get(content_id, 0))
            watch = 1 if watch_time > POSITIVE_WATCH_THRESHOLD else 0
            watch_2m = 1 if watch_time > 120 else 0
            paywall_view = (
                1 if paywall_view_final_ts.get(content_id, 0) > timestamp else 0
            )
            add_watchlist = (
                1 if watchlist_add_final_ts.get(content_id, 0) > timestamp else 0
            )
            positives.append(
                MtlExample(
                    tray_id=tray_id,
                    content_id=content_id,
                    timestamp=first_impr_ts,
                    click=1,
                    watch=watch,
                    watch_time=watch_time,
                    paywall_view=paywall_view,
                    add_watchlist=add_watchlist,
                )
            )
            visited_content_ids.add(content_id)

        for tray_id, agg in tray_id_to_agg.items():
            for content_id, ts in agg.impressions.items():
                if (
                    content_id in visited_content_ids
                    or content_id in clicked_content_ids
                    or content_id in today_content_watch_time
                ):
                    continue

                negatives.append(
                    MtlExample(
                        tray_id=tray_id,
                        content_id=content_id,
                        timestamp=ts,
                        click=0,
                        watch_time=0,
                        watch=0,
                        paywall_view=0,
                        add_watchlist=0,
                    )
                )
                visited_content_ids.add(content_id)

        if len(positives) == 0 or len(negatives) == 0:
            return []
        negatives = [ex for ex in negatives if random.random() < uni_ns_rate]
        return positives + negatives

    return generate_mtl_examples_v3


def generate_daily_impr_examples(
    spark,
    date,
    dw_pid_filter,
    country,
    bc_convert_to_show_content_id_map,
    mtl_uni_ns_rate,
):
    daily_example_path = (
        data_path(TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_IMPR_EXAMPLES, country) % date
    )
    if is_s3_path_success(daily_example_path):
        print(f"use existing daily impr examples on {date}")
        return

    print(f"generate impression examples in date {date}")

    impressions = load_daily_tray_impression(spark, date, dw_pid_filter, country)
    complete_watches = load_raw_complete_ent_watches(
        spark, date, dw_pid_filter, country
    )

    watches = load_daily_tray_ent_watch(spark, date, dw_pid_filter, country)

    past_raw_complete_watches = load_raw_complete_ent_watches(
        spark, get_diff_date(date, -1), dw_pid_filter, country
    )
    ent_watches = load_daily_ent_watch(spark, date, dw_pid_filter, country)

    clicks = load_daily_tray_tile_click(spark, date, dw_pid_filter, country)
    watchlist_adds = load_daily_add_watchlist(spark, date, dw_pid_filter, country)
    paywall_views = load_daily_paywall_view(spark, date, dw_pid_filter, country)

    print("daily example path", daily_example_path)
    udf_generate_samples = F.udf(
        generate_user_samples_v3, returnType=ArrayType(ImprExampleRowType)
    )
    tpfy_examples = (
        impressions.join(watches, on="dw_p_id", how="inner")
        .join(complete_watches, on="dw_p_id", how="left")
        .select(
            "dw_p_id",
            udf_generate_samples(
                "tpfy_impressions", "tray_ent_watches", "complete_watches"
            ).alias("tpfy_examples"),
        )
        .filter(F.size("tpfy_examples") > 0)
    )

    print("mtl uniform negative sampling rate", mtl_uni_ns_rate)
    udf_generate_mtl_examples = F.udf(
        generate_mtl_examples_v3_builder(
            bc_convert_to_show_content_id_map, uni_ns_rate=mtl_uni_ns_rate
        ),
        returnType=ArrayType(MtlExampleRowType),
    )
    tpfy_mtl_examples = (
        impressions.join(clicks, on="dw_p_id", how="inner")
        .join(ent_watches, on="dw_p_id", how="left")
        .join(complete_watches, on="dw_p_id", how="left")
        .join(watchlist_adds, on="dw_p_id", how="left")
        .join(paywall_views, on="dw_p_id", how="left")
        .select(
            "dw_p_id",
            udf_generate_mtl_examples(
                "tpfy_impressions",
                "watches",
                "complete_watches",
                "tile_clicks",
                "paywall_views",
                "watchlist_adds",
            ).alias("tpfy_mtl_examples"),
        )
        .filter(F.size("tpfy_mtl_examples") > 0)
    )

    udf_generate_lastwatch_examples = F.udf(
        generate_lastwatch_examples, returnType=ArrayType(ImprExampleRowType)
    )
    lastwatch_examples = (
        ent_watches.join(past_raw_complete_watches, on="dw_p_id", how="left")
        .select(
            "dw_p_id",
            udf_generate_lastwatch_examples("watches", "complete_watches").alias(
                "lastwatch_examples"
            ),
        )
        .filter(F.size("lastwatch_examples") > 0)
    )

    examples = tpfy_examples.join(lastwatch_examples, on="dw_p_id", how="full").join(
        tpfy_mtl_examples, on="dw_p_id", how="full"
    )

    # print("examples count", examples.count())
    remove_s3_folder(daily_example_path)
    examples.repartition(8, "dw_p_id").write.mode("overwrite").parquet(
        daily_example_path
    )

    def arraysize(arr):
        return F.when(F.size(arr) > 0, F.size(arr)).otherwise(F.lit(0))

    def nonempty(arr):
        return (F.size(arr) > 0).cast(IntegerType())

    res = spark.read.parquet(daily_example_path)
    stat = res.agg(
        F.count(F.lit(1)).alias("total"),
        F.sum(nonempty("tpfy_examples")).alias("tpfy_nu"),
        F.sum(nonempty("lastwatch_examples")).alias("lw_nu"),
        F.sum(arraysize("tpfy_examples")).alias("tpfy_ne"),
        F.sum(arraysize("tpfy_mtl_examples")).alias("tpfy_mtl_ne"),
    ).collect()
    print("stat", stat)
