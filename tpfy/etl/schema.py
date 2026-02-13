import numpy as np

from model.schema import (
    Feature,
    DataSchema,
)


class _TpfyDatasetSchemaV0:
    _meta_output = [
        Feature("secs_start_dt", np.int32, (), 0),
        Feature("flag", np.int64, (), np.int64(0)),
        Feature("country", np.str_, (), ""),
        Feature("dw_p_id", np.str_, (), ""),
        Feature("content_id", np.str_, (), ""),
        Feature("timestamp", np.int64, (), np.int64(0)),
    ]

    _label_output = [
        Feature("label", np.int32, (1,), 0),
    ]

    # not read/required in training
    _raw_feature_output = []

    # new features should be added as fid
    _fid_output = [
        Feature(
            "user_fids",
            np.int64,
            (None,),
            np.int64(0),
            as_model_input=True,
            compact_predict=True,
        ),
        Feature(
            "user_weighted_fids",
            np.int64,
            (None,),
            np.int64(0),
            as_model_input=True,
            compact_predict=True,
        ),
        Feature(
            "user_weighted_fid_weights",
            np.float32,
            (None,),
            0.0,
            as_model_input=True,
            compact_predict=True,
        ),
        Feature("fids", np.int64, (None,), np.int64(0), as_model_input=True),
        Feature("weighted_fids", np.int64, (None,), np.int64(0), as_model_input=True),
        Feature("weighted_fid_weights", np.float32, (None,), 0.0, as_model_input=True),
        Feature(
            "sparse_indices",
            np.int32,
            (None,),
            0.0,
            as_model_input=True,
            compact_predict=True,
        ),
        Feature("sparse_values", np.float32, (None,), 0.0, as_model_input=True),
        Feature("task", np.int32, (1,), 0, as_model_input=True, compact_predict=True),
    ]

    # for etl job
    all_output = _meta_output + _label_output + _raw_feature_output + _fid_output

    # for model training
    features = [o for o in _fid_output if o.as_model_input] + []
    labels = _label_output
    metadata = _meta_output


TpfyDatasetSchema = DataSchema(
    "TpfyDatasetSchema",
    _TpfyDatasetSchemaV0.all_output,
    _TpfyDatasetSchemaV0.features,
    _TpfyDatasetSchemaV0.labels,
    _TpfyDatasetSchemaV0.metadata,
)


class _TpfyDatasetSchemaMTL:
    _meta_output = [
        Feature("secs_start_dt", np.int32, (), 0),
        Feature("flag", np.int64, (), np.int64(0)),
        Feature("country", np.str_, (), ""),
        Feature("dw_p_id", np.str_, (), ""),
        Feature("content_id", np.str_, (), ""),
        Feature("timestamp", np.int64, (), np.int64(0)),
    ]

    _label_output = [
        Feature("click", np.int32, (1,), 0),
        Feature("watch", np.int32, (1,), 0),
        Feature("paywall_view", np.int32, (1,), 0),
        Feature("add_watchlist", np.int32, (1,), 0),
    ]

    # not read/required in training
    _raw_feature_output = []

    # new features should be added as fid
    _fid_output = [
        Feature(
            "user_fids",
            np.int64,
            (None,),
            np.int64(0),
            as_model_input=True,
            compact_predict=True,
        ),
        Feature(
            "user_weighted_fids",
            np.int64,
            (None,),
            np.int64(0),
            as_model_input=True,
            compact_predict=True,
        ),
        Feature(
            "user_weighted_fid_weights",
            np.float32,
            (None,),
            0.0,
            as_model_input=True,
            compact_predict=True,
        ),
        Feature("fids", np.int64, (None,), np.int64(0), as_model_input=True),
        Feature("weighted_fids", np.int64, (None,), np.int64(0), as_model_input=True),
        Feature("weighted_fid_weights", np.float32, (None,), 0.0, as_model_input=True),
        Feature(
            "sparse_indices",
            np.int32,
            (None,),
            0.0,
            as_model_input=True,
            compact_predict=True,
        ),
        Feature("sparse_values", np.float32, (None,), 0.0, as_model_input=True),
        Feature("task", np.int32, (1,), 0, as_model_input=True, compact_predict=True),
    ]

    # for etl job
    all_output = _meta_output + _label_output + _raw_feature_output + _fid_output

    # for model training
    features = [o for o in _fid_output if o.as_model_input] + []
    labels = _label_output
    metadata = _meta_output


TpfyMtlDatasetSchema = DataSchema(
    "TpfyMtlDatasetSchema",
    _TpfyDatasetSchemaMTL.all_output,
    _TpfyDatasetSchemaMTL.features,
    _TpfyDatasetSchemaMTL.labels,
    _TpfyDatasetSchemaMTL.metadata,
)
