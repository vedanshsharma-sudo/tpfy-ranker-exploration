from tpfy.tf_dataset.feature import XDeepFMFeatureExtractor
import numpy as np
import functools

var_names = [
    "task_bias",
    "watched_embeddings",
    "language_embeddings",
    "genre_embeddings",
    "production_house_embeddings",
    "entitlement_embeddings",
    "parental_rating_embeddings",
    "studio_embeddings",
    "content_type_embeddings",
    "year_bucket_embeddings",
    "sports_language_embeddings",
    "sports_watched_embeddings",
    "age_embeddings",
    "gender_embeddings",
    "joined_embeddings",
    "sub_plan_embeddings",
    "target_id_embeddings",
    "target_language_embeddings",
    "target_genre_embeddings",
    "target_production_house_embeddings",
    "target_entitlement_embeddings",
    "target_parental_rating_embeddings",
    "target_studio_embeddings",
    "target_content_type_embeddings",
    "target_year_bucket_embeddings",
    "location_embeddings",
    "device_embeddings",
    "wide_features",
]

tensor_names = [
    "exDeepFm_9/GatherV2:0",
    "Reshape_34:0",
    "Reshape_35:0",
    "Reshape_36:0",
    "Reshape_37:0",
    "Reshape_38:0",
    "Reshape_39:0",
    "Reshape_40:0",
    "Reshape_41:0",
    "Reshape_42:0",
    "mul_32:0",
    "mul_31:0",
    "Mean_35:0",
    "Mean_36:0",
    "Mean_37:0",
    "Mean_38:0",
    "Mean_26:0",
    "Mean_27:0",
    "Mean_28:0",
    "Mean_29:0",
    "Mean_30:0",
    "Mean_31:0",
    "Mean_32:0",
    "Mean_33:0",
    "Mean_34:0",
    "AddN_4:0",
    "AddN_5:0",
    "wide_features:0",
]


def get_baseline_input(batch_size=1, **kwargs):
    batch_data = []
    for name, tf_type, shape, pad_value in XDeepFMFeatureExtractor.output:
        if len(shape) == 1 and shape[0] is None:
            shape = [1]
        size = batch_size * functools.reduce(lambda a, b: a * b, shape)
        if name.endswith("weights"):
            pad_value = 1.0
        value = [pad_value] * size
        batch_data.append(np.reshape(value, (batch_size,) + tuple(shape)))
    return batch_data
