import numpy as np

from model.schema import (
    Column,
    DataSchema,
)


class _CumWatchSchema:
    _meta_output = [
        Column("user_hash", np.int64, (), 0),
    ]

    _label_output = [
        Column("cum_watch_time", np.float32, (), 0.0),
        Column("all_watch_time", np.float32, (), 0.0),
    ]

    _raw_feature_output = [
        Column("dw_p_id", np.str_, (), ""),
        Column("timestamp", np.int64, (), np.int64(0)),
        Column("content_id", np.int64, (), np.int64(0)),
        Column("raw", np.float32, (None,), 0.0),
    ]

    # new features should be added as fid
    _fid_output = [
        Column("day", np.int32, (), 0, as_model_input=True),
        Column(
            "user_fids",
            np.int64,
            [None],
            np.int64(0),
            as_model_input=True,
            compact_predict=True,
        ),
        Column(
            "user_weighted_fids",
            np.int64,
            [None],
            np.int64(0),
            as_model_input=True,
            compact_predict=True,
        ),
        Column(
            "user_weighted_fid_weights",
            np.float32,
            [None],
            0.0,
            as_model_input=True,
            compact_predict=True,
        ),
        Column("fids", np.int64, (None,), np.int64(0), as_model_input=True),
        Column("weighted_fids", np.int64, (None,), np.int64(0), as_model_input=True),
        Column("weighted_fid_weights", np.float32, (None,), 0.0, as_model_input=True),
    ]

    # for etl job
    all_output = _meta_output + _label_output + _raw_feature_output + _fid_output

    # for model training
    features = [o for o in _fid_output if o.as_model_input] + []
    labels = _label_output
    metadata = _meta_output


CumWatchDataSchema = DataSchema(
    "CumWatchSchemaV0",
    _CumWatchSchema.all_output,
    _CumWatchSchema.features,
    _CumWatchSchema.labels,
    _CumWatchSchema.metadata,
)
