from collections import namedtuple

DeviceInfo = namedtuple(
    "DeviceInfo",
    [
        "ts",
        "platform",
        "country",
        "state",
        "city",
        "carrier",
        "asn_number",
        "manufacturer",
        "screen_height",
        "model",
    ],
)


class TpfyDataPath:
    S3_CUMULATIVE_WATCH_TRAIN_BASE = "tpfy_cum_watch_datasets/base/cd=%s"
    S3_CUMULATIVE_WATCH_TRAIN_BASE_WATCHED = (
        "tpfy_cum_watch_datasets/base-watched/cd=%s"
    )
    S3_CUMULATIVE_WATCH_TRAIN_METADATA = "tpfy_cum_watch_datasets/metadata_v2/%s"
    S3_CUMULATIVE_WATCH_TRAIN_METADATA_CMS3 = "tpfy_cum_watch_datasets/metadata_cms3/%s"
    S3_CUMULATIVE_WATCH_TRAIN_FEATURE = "tpfy_cum_watch_datasets/feature%s/cd=%s"
    S3_CUMULATIVE_WATCH_TRAIN_EXAMPLE_D7 = "tpfy_cum_watch_datasets/example-d7%s/cd=%s"

    S3_CUMULATIVE_WATCH_TRAIN_EXAMPLE_PROG = (
        "tpfy_cum_watch_datasets/example-prog%s/cd=%s"
    )

    S3_CUMULATIVE_WATCH_MODEL_EXPORT = "tpfy-cum-watch/%s"

    S3_TPFY_IMPR_V3_DAILY_IMPR_EXAMPLES = "tpfy-impr-v3/daily-mtl/%s"
    S3_TPFY_IMPR_V3_DAILY_RAW = "tpfy-impr-v3/daily-mtl-raw/%s"
    S3_TPFY_IMPR_V3_DAILY_EXTRACTED_EXAMPLES = "tpfy-impr-v3/daily-extracted%s/cd=%s"
    S3_TPFY_IMPR_V3_DAILY_MTL_EXTRACTED_EXAMPLES = (
        "tpfy-impr-v3/daily-mtl-extracted%s/cd=%s"
    )

    S3_TPFY_IMPR_V3_DAILY_METADATA_CACHE = "tpfy-impr-v3/metadata/%s/metadata.pkl"
    S3_TPFY_IMPR_V3_DAILY_METADATA_CACHE_CMS3 = (
        "tpfy-impr-v3/metadata-cms3/%s/metadata.pkl"
    )
    S3_TPFY_IMPR_V3_AGG_EXTRACTED_EXAMPLES_VAR = "tpfy-impr-v3/agg-extracted%s/%s"
    S3_TPFY_IMPR_V3_AGG_MTL_EXTRACTED_EXAMPLES_VAR = (
        "tpfy-impr-v3/agg-mtl-extracted%s/%s"
    )

    S3_TPFY_MODEL_EXPORT = "tpfy/%s"
    
    S3_TPFY_NEURAL_LINUCB_MATRICES = "tpfy/tpfy-v3-neural-linucb"