import os
import pickle
import time
import json
from operator import itemgetter
from common.s3_utils import upload_string, download_single_file, upload_file
from pyspark.sql.functions import (
    col,
    udf,
    count,
    collect_list,
    explode,
    array as _array,
    when,
    size,
    array_except,
)
from pyspark.sql.types import (
    BooleanType,
    StructType,
    StructField,
    StringType,
    LongType,
    ArrayType,
)

from common.cms3.cms3_utils import get_cms_store
from common.constants5 import (
    CNT_ID_TYPE,
    S3_AP_BASE_PATH,
)
from common.dw_mapping_utils import map_key_to_pid, check_and_alert_mappings
from common.data.behavior.watch import Watch2
from common.s3_utils import is_s3_path_success
from common.spark_utils import load_user_profile
from common.time_utils import (
    timestamp,
    get_dates_str_backwards,
    get_dates_list_backwards,
)
from common.utils import is_premium, boosting_pay_candidates
from common.config import TENANT
from common.config.utils import data_path, get_config, tenant_countries
from common.config.constants import DataPath, EMPTY_USER_DW_PID
from tpfy.dao.tpfy_dynamo_dao import TFPYDDynamoDB
from tpfy.emr_prediction.processers import filter_by_language
from tpfy.tf_model.retriver.item_based_retriever import ItemBasedRetriever
from datetime import datetime


VALID_CONTENT_TYPES = set(CNT_ID_TYPE.keys())


class PredictionStats:
    def __init__(self, spark_context):
        self.predict_count = spark_context.accumulator(0)
        self.empty_count = spark_context.accumulator(0)
        self.total_result_count = spark_context.accumulator(0)

    def accumulate(self, predicted, empty, total):
        self.predict_count.add(predicted)
        self.empty_count.add(empty)
        self.total_result_count.add(total)

    def pretty(self):
        return (
            f"predict_count: {self.predict_count}\n"
            f"empty_count: {self.empty_count}\n"
            f"total_result_count: {self.total_result_count}\n"
        )


def load_meta_data(sql_context, movies, shows, date_str, now, country):
    def is_alive(content_id):
        return (
            content_id in movies
            and movies[content_id].content_type_id == 200
            and (not movies[content_id].deleted)
            and (not movies[content_id].hidden)
        ) or (
            content_id in shows
            and (not shows[content_id].deleted)
            and (not shows[content_id].hidden)
        )

    filtered_cids = {}
    raw_relevances = sql_context.read.csv(
        data_path(DataPath.S3_ITEM_RELEVANCES, TENANT) % date_str
    ).rdd.collect()

    ent_relevances = {}
    for cid, lang, dst_str in raw_relevances:
        key = (int(cid), int(lang))
        tks = dst_str.split(",")
        dsts = []
        for i in range(0, len(tks), 3):
            dst_id = int(tks[i])
            if is_alive(dst_id):
                dsts.append(((dst_id, int(tks[1 + i])), float(tks[2 + i])))
            else:
                filtered_cids[dst_id] = 1 + filtered_cids.get(dst_id, 0)
        ent_relevances[key] = dsts
    print("Loaded Item Relevances", len(ent_relevances))

    print("========== filtered Dst Contents ==========")
    print(",".join([str(cid) for cid in filtered_cids.keys()]))
    print(
        [
            (cid, count, cid in movies or cid in shows)
            for cid, count in sorted(
                filtered_cids.items(), key=itemgetter(1), reverse=True
            )
        ]
    )
    return ent_relevances


def load_user_data(sql_context, args, country):
    dw_pid_filter = udf(lambda t: True, returnType=BooleanType())
    print("pid subset empty, predict for all user")

    @udf
    def get_epoch_days(timestamp):
        return timestamp // (24 * 60 * 60)

    watches = (
        sql_context.read.option("mergeSchema", "false")
        .parquet(data_path(DataPath.S3_GROUP_WATCH_UBS, country) % args.date_str)
        .filter(col("dw_p_id").isNotNull() & (col("dw_p_id") != EMPTY_USER_DW_PID))
        .select(["dw_p_id", "watches"])
    )

    user_dates = get_dates_str_backwards(args.date_str, args.days)
    users = (
        sql_context.read.option("mergeSchema", "false")
        .parquet(data_path(DataPath.S3_DAILY_WATCH_ENT, country) % user_dates)
        .select(["dw_p_id"])
        .filter(col("dw_p_id").isNotNull())
        .filter(col("dw_p_id") != "")
        .filter(col("dw_p_id") != EMPTY_USER_DW_PID)
        .filter(dw_pid_filter("dw_p_id"))
    )

    if not args.exclude_sports_and_news:
        users_ns = (
            sql_context.read.option("mergeSchema", "false")
            .parquet(
                data_path(DataPath.S3_DAILY_WATCH_NEWS_SPORTS, country) % user_dates
            )
            .select(["dw_p_id"])
            .filter(col("dw_p_id").isNotNull())
            .filter(col("dw_p_id") != "")
            .filter(col("dw_p_id") != EMPTY_USER_DW_PID)
            .filter(dw_pid_filter("dw_p_id"))
        )
        users = users.union(users_ns)

    users = users.distinct()
    watches = watches.join(users, on="dw_p_id", how="inner")
    watches = watches.select(
        [
            "dw_p_id",
            "watches",
        ]
    ).rdd.map(lambda t: (t[0], t[1:]))

    return watches


def is_show(key, shows):
    return key[0] in shows


def is_movie(key, movies):
    return key[0] in movies and movies[key[0]].content_type_id == 200


def is_disneyplus(key, movies, shows):
    return (key[0] in movies and movies[key[0]].is_disneyplus) or (
        key[0] in shows and shows[key[0]].is_disneyplus
    )


def filter_results(results, watched, filter_fn=None):
    frs, visited = [], set()
    for (cid, lang), score in results:
        if cid in watched or cid in visited:
            continue
        visited.add(cid)
        if filter_fn is None or filter_fn((cid, lang)):
            frs.append((cid, score))
    return frs


def finalize_results(
    results, result_count, min_pay_ratio, is_premium_fn, cid_map_fn=None
):
    boosting_pay_scores = boosting_pay_candidates(
        results, result_count, min_pay_ratio, is_premium_fn
    )

    frs = []
    for cid, score in boosting_pay_scores:
        if cid_map_fn is not None:
            frs.append((cid_map_fn(cid), score))
        else:
            frs.append((cid, score))
    return frs


def predict(
    user_watches,
    broadcast_metadata,
    ent_relevances,
    prediction_start,
    args,
    accumulator=None,
    watch_len_thres=600,
):
    movies, shows = broadcast_metadata.value
    retriever = ItemBasedRetriever(ent_relevances.value)
    predicted, empty, total = 0, 0, 0

    parse_watch_fn = Watch2.parse_watches_fn()

    for p_id, (raw_ent_watches,) in user_watches:
        if len(p_id) > 1000:
            continue

        ent_watches = parse_watch_fn(raw_ent_watches)
        item_based_candidates, evidences, weights = retriever.retrieve(
            ent_watches,
            prediction_start,
            shows,
            movies,
            watch_len_thres=watch_len_thres,
        )

        results = [((c, l), s) for (c, l), s in item_based_candidates]
        results = filter_by_language(ent_watches, results)
        results.sort(key=itemgetter(1), reverse=True)

        # post process
        watched = set(watch.content_id for watch in ent_watches)
        home_results = filter_results(
            list(results),
            watched,
            filter_fn=lambda k: is_show(k, shows) or is_movie(k, movies),
        )
        final_home_results = finalize_results(
            home_results,
            args.result_count,
            args.min_pay_ratio,
            is_premium_fn=lambda cid: is_premium(cid, movies, shows),
        )

        if predicted % 1000 == 0:
            print(f"{datetime.utcnow()}: {predicted} prediction finished, ")
        yield p_id, final_home_results

        predicted += 1
        total += len(results)
        if len(results) == 0:
            empty += 1

    if accumulator is not None:
        accumulator.accumulate(predicted, empty, total)


def extract_predictions(row):
    pid, row_str = row
    row_json = json.loads(row_str)
    return (
        pid,
        [int(cell["S"]) for cell in row_json["result"]["L"]],
    )


def upload_to_dynamo(user_results):
    dynamo_db = TFPYDDynamoDB(os.environ["DYNAMO_TABLE"])
    return dynamo_db.batch_put_generator(
        user_results, pack_fn=TFPYDDynamoDB.encode_item
    )


MOVIE_FIELDS = ["content_type_id", "premium", "is_disneyplus", "hidden", "deleted"]
TV_SHOW_FIELDS = ["content_type_id", "premium", "is_disneyplus", "hidden", "deleted"]


def run_prediction(spark_context, sql_context, args, country):
    spark_context.setCheckpointDir(
        data_path(DataPath.S3_TPFY_UPLOAD_CACHE, country) % "checkpoint"
    )
    prediction_start = timestamp(args.date_str) + 86400

    cms_store = get_cms_store(
        sql_context=sql_context,
        country_list=[country],
        movie_fields=MOVIE_FIELDS,
        tv_show_fields=TV_SHOW_FIELDS,
        use_mock_table=False,
        return_tuples=False,
    )
    movies = dict(cms_store.movies.get_items(country))
    shows = dict(cms_store.tv_shows.get_items(country))

    contents_bc = spark_context.broadcast((movies, shows))
    data = load_meta_data(
        sql_context, movies, shows, args.date_str, prediction_start, country
    )
    data_bc = spark_context.broadcast(data)

    watches = load_user_data(sql_context, args, country)

    received_at = timestamp(args.date_str) - (30 + args.days) * 86400
    watches, total, non_pid_device, non_pid = map_key_to_pid(
        spark_context,
        sql_context,
        watches,
        received_at=received_at,
        countries=[country],
    )
    watches.checkpoint()
    print("Users to predict: ", watches.count())

    accumulator = PredictionStats(spark_context)
    predictions = watches.mapPartitions(
        lambda user_watches: predict(
            user_watches,
            contents_bc,
            data_bc,
            prediction_start,
            args,
            accumulator=accumulator,
            watch_len_thres=600,
        ),
        preservesPartitioning=True,
    )
    # Save results to upload
    s3_upload_path = data_path(DataPath.S3_TPFY_UPLOAD_CACHE, country) % args.date_str
    predictions.map(
        lambda rs: (rs[0], json.dumps(TFPYDDynamoDB.pack_item(*rs, decode=True)))
    ).filter(lambda rs: rs[1] is not None and rs[1] != "null").toDF(
        ["pid", "row_str"]
    ).write.mode(
        "overwrite"
    ).parquet(
        s3_upload_path
    )

    # Save predictions
    checkpoint = int(time.time())
    s3_path = data_path(DataPath.S3_TPFY_PREDICTION, country, "") % (
        args.date_str,
        "als",
        checkpoint,
        1,
    )
    sql_context.read.parquet(s3_upload_path).select(["pid", "row_str"]).rdd.map(
        extract_predictions
    ).toDF(
        schema=StructType(
            [
                StructField("p_id", StringType(), False),
                StructField("predictions", ArrayType(LongType()), False),
            ]
        )
    ).write.mode(
        "overwrite"
    ).parquet(
        s3_path
    )

    upload_string(
        data_path(DataPath.S3_NOTIFICATION_TPFY_FINISH, country, "") % args.date_str
        + "/_SUCCESS",
        s3_path,
    )
    print(accumulator.pretty())

    # Export to AP
    if args.export_to_ap:
        s3_path = data_path(DataPath.S3_TPFY_PREDICTION, country, "") % (
            args.date_str,
            "als",
            checkpoint,
            1,
        )
        ap_path = os.path.join(
            S3_AP_BASE_PATH, country.upper(), "tpfy", "cd=%s", "%s"
        ) % (args.date_str, "als")
        print("export tpfy results from %s to %s" % (s3_path, ap_path))
        time.sleep(60)  # wait for eventually consistency of S3
        ap_df = sql_context.read.parquet(s3_path)
        ufs_path = next(
            iter(
                [
                    data_path(DataPath.S3_USER_FEATURE, country) % d
                    for d in get_dates_list_backwards(args.date_str, 7)
                    if is_s3_path_success(
                        data_path(DataPath.S3_USER_FEATURE, country) % d
                    )
                ]
            ),
            None,
        )
        if ufs_path:
            ufs_df = (
                sql_context.read.option("mergeSchema", "false")
                .parquet(ufs_path)
                .withColumn("latest_watches", col("content_ids.content_id"))
                .join(
                    load_user_profile(sql_context)
                    .select("dw_p_id", "pid")
                    .filter(col("pid").isNotNull()),
                    "dw_p_id",
                    "inner",
                )
                .withColumnRenamed("pid", "p_id")
                .select("p_id", "latest_watches")
            )
            ap_df = ap_df.join(ufs_df, "p_id", "left")
            for pred in [
                "predictions",
            ]:
                ap_df = ap_df.withColumn(
                    pred,
                    when(
                        size("latest_watches") > 0,
                        array_except(col(pred), col("latest_watches")),
                    ).otherwise(col(pred)),
                )
            ap_df = ap_df.drop("latest_watches")
        ap_df.write.mode("overwrite").parquet(ap_path)

    total_ids = total.value
    missing_from_device = non_pid_device.value
    missing_all = non_pid.value
    check_and_alert_mappings(total_ids, missing_from_device, missing_all)


def run_upload(sql_context, args, country):
    s3_upload_path = data_path(DataPath.S3_TPFY_UPLOAD_CACHE, country) % args.date_str
    predictions = (
        sql_context.read.parquet(s3_upload_path)
        .select(["pid", "row_str"])
        .rdd.values()
        .map(lambda x: json.loads(x))
        .mapPartitions(upload_to_dynamo, preservesPartitioning=True)
    )
    print("upload count:", predictions.count())
    upload_string(
        data_path(DataPath.S3_TPFY_UPLOAD_CACHE, country)
        % ("uploaded/" + args.date_str),
        "SUCCESS",
    )
