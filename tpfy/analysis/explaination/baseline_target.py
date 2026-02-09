from tpfy.tf_dataset.feature import XDeepFMFeatureExtractor
import numpy as np
import functools


var_names = [
    "target_id_embeddings",
    "target_language_embeddings",
    "target_genre_embeddings",
    "target_production_house_embeddings",
    "target_entitlement_embeddings",
    "target_parental_rating_embeddings",
    "target_studio_embeddings",
    "target_content_type_embeddings",
    "target_year_bucket_embeddings",
    "wide_features",
]

tensor_names = [
    "Mean_26:0",
    "Mean_27:0",
    "Mean_28:0",
    "Mean_29:0",
    "Mean_30:0",
    "Mean_31:0",
    "Mean_32:0",
    "Mean_33:0",
    "Mean_34:0",
    "wide_features:0",
]


def get_baseline_input(batch_size=1, **kwargs):
    batch_data = []
    for cell in kwargs["meta_feature"] + kwargs["user_feature"]:
        batch_data.append([cell for _ in range(batch_size)])

    for name, tf_type, shape, pad_value in (
        XDeepFMFeatureExtractor._target_output + XDeepFMFeatureExtractor._wide_output
    ):
        if len(shape) == 1 and shape[0] is None:
            shape = [1]
        size = batch_size * functools.reduce(lambda a, b: a * b, shape)
        if name.endswith("weights"):
            pad_value = 1.0
        value = [pad_value] * size
        batch_data.append(np.reshape(value, (batch_size,) + tuple(shape)))
    return batch_data
