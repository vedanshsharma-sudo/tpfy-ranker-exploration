from datetime import datetime, timedelta
import os

import pyspark.sql.functions as F

from common.config import REGION
from common.config.constants import RECO_S3_BUCKET
from model.emr_utils import get_daily_event_partitions


def to_str(value):
    if value is not None:
        return str(value)


def date_to_ts(d):
    nxt = datetime(d.year, d.month, d.day) + timedelta(days=1)
    return int(nxt.timestamp())


def load_tpfy_predict_events(sql_context, date, country):
    if country == "in":
        event_key = "in"
    elif REGION.startswith("apse"):
        event_key = "sea"
    elif REGION.startswith("euw"):
        event_key = "mea"
    else:
        raise Exception(f"unexpected event region {REGION}")

    base_path = os.path.join(RECO_S3_BUCKET, f"events/tpfy-v2/predict/{event_key}")
    path = os.path.join(base_path, get_daily_event_partitions(date, country))
    print("load tpfy events raw feature events from:", path)
    events = (
        sql_context.read.option("basePath", base_path)
        .parquet(path)
        .filter(F.length("payload") > 0)
    )
    if country != "in":
        events = events.filter(F.col("country") == country)

    agg_events = events.groupBy("dw_p_id").agg(
        F.collect_list(
            F.struct("timestamp", "platform", "subscription_types", "payload")
        ).alias("events")
    )
    return agg_events
