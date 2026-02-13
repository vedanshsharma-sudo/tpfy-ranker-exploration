import os

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
import tensorflow_recommenders_addons as tfra
import tensorflow_hs_addon as tfhs

from common.config.utils import get_config
from common.config import TENANT
from tpfy.tf_dataset.feature import (
    schema,
)
from model.fid import feature_bits
from model.model_utils import FeatureGroup
from tray_ranking.train.layers import DNN
from tensorflow.keras.layers import Dense

NO_USER_IDENTITY = get_config(TENANT)["general"]["NO_USER_IDENTITY"]


def fm(user_embeddings, target_embeddings):
    # user_embeddings: [1 or N, m, k]
    # target_embeddings: [N, n, k]
    m = user_embeddings.shape[1].value
    n = target_embeddings.shape[1].value

    if m is None or n is None:
        raise Exception("num feature dimension must be known in compile time")

    inner_products = tf.matmul(
        user_embeddings, target_embeddings, transpose_b=True
    )  # N, m, n
    return tf.reshape(inner_products, [-1, m * n])


class CumWatchModel(tf.keras.Model):
    def __init__(self, dim, use_featuredump, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.embedding_l2 = 1e-3

        self.use_featuredump = use_featuredump

        self.dynamic_embedding_var = tfra.dynamic_embedding.get_variable(
            name="embedding_layer",
            key_dtype=tf.int64,
            value_dtype=tf.float32,
            dim=dim,
            initializer=tf.initializers.random_uniform(),
            bp_v2=True,
        )

        self.de_trainable_wrappers = []
        self.de_vars = [self.dynamic_embedding_var]

        self.dnn = DNN(
            hidden_units=[256, 128], bias_initializer=tf.initializers.constant(0.1)
        )
        self.linear = Dense(7)

    def get_feature_embedding(
        self,
        fids,
        weighted_fids,
        weighted_fid_weights,
        slots_to_pool,
        name,
        is_training,
        cid_dropout,
    ) -> FeatureGroup:
        with tf.name_scope(f"{name}_feature"):
            fid_weights = tf.concat(
                [tf.ones_like(fids, dtype=tf.float32), weighted_fid_weights],
                axis=1,
                name="fid_weight_concat",
            )
            fids = tf.concat([fids, weighted_fids], axis=1, name="fid_concat")

            if is_training and cid_dropout:
                true_fid_slots = tf.cast(
                    tf.bitwise.right_shift(fids, feature_bits),
                    tf.int32,
                    name="fid_slots",
                )
                user_should_dropout = tf.random.uniform([tf.shape(fids)[0]]) < 0.1
                is_cid = tf.equal(true_fid_slots, schema.target_id.slot_id)
                should_dropout = tf.logical_and(
                    user_should_dropout[:, tf.newaxis], is_cid
                )
                fids = tf.where(should_dropout, tf.constant(0, tf.int64), fids)

            fid_slots = tf.cast(
                tf.bitwise.right_shift(fids, feature_bits), tf.int32, name="fid_slots"
            )
            fids = tf.where(tfhs.isin(fid_slots, slots_to_pool), fids, 0)

            flat_fids = tf.reshape(fids, (-1,))
            unique_fids, flat_indices = tf.unique(flat_fids, name="fid_dedup")
            fid_indices = tf.reshape(flat_indices, tf.shape(fids), name="fid_indices")

            if is_training:
                unique_embeddings, wrapper = tfra.dynamic_embedding.embedding_lookup(
                    self.dynamic_embedding_var,
                    unique_fids,
                    return_trainable=True,
                    name=f"de_lookup",
                )

                if self.embedding_l2 > 0:
                    embedding_reg_loss = tf.identity(
                        self.embedding_l2
                        * tf.reduce_sum(unique_embeddings**2, keepdims=False),
                        name=f"{name}_embedding_reg_loss",
                    )
                    self.add_loss(embedding_reg_loss)
                self.de_trainable_wrappers.append(wrapper)
                missed_fids = None
            else:
                unique_embeddings, exists = self.dynamic_embedding_var.lookup(
                    unique_fids, return_exists=True, name="de_lookup"
                )
                missed_fids = unique_fids[~exists]
                mask = tf.cast(exists, tf.float32)
                unique_embeddings = tf.identity(
                    unique_embeddings * mask[:, np.newaxis], name="clear_unknown_fid"
                )

            pooled_embeddings, _ = tfhs.pooling_by_slots(
                fid_indices, fid_slots, fid_weights, unique_embeddings, slots_to_pool
            )
            pooled_embeddings = tf.identity(
                pooled_embeddings, name=f"{name}_embeddings"
            )

            fg = FeatureGroup()
            fg.fids = fids
            fg.fid_weights = fid_weights
            fg.fid_indices = fid_indices
            fg.fid_slots = fid_slots
            fg.slot_id_to_index = {sid: i for i, sid in enumerate(slots_to_pool)}
            fg.unique_embeddings = unique_embeddings
            fg.pooled_embeddings = pooled_embeddings
            return fg

    def call(self, inputs, training=False, compact_predict=False):
        user_slots_v1 = [
            schema.raw_plan_types,
            schema.gender,
            schema.age,
            schema.watched_content,
            schema.watched_entitlements,
            schema.watched_language_id,
            schema.watched_studio_id,
            schema.watched_genre_id,
            schema.watched_content_type,
            schema.online_platform,
            schema.state,
            schema.asn_number,
            schema.manufacturer,
            schema.screen_height,
            schema.is_paid_user,
            schema.is_honeypot_user,
            schema.is_honeypot_enabled,
        ]

        if self.use_featuredump:
            user_slots_v1.append(schema.total_watch_hours_bucket)

        item_slots_v1 = [
            schema.target_id,
            schema.target_content_type_id,
            schema.target_genre_id,
            schema.target_studio_id,
            schema.target_entitlement,
            schema.target_parental_rating_id,
            schema.target_channel_id,
            schema.target_language_id,
        ]

        user_slots = user_slots_v1
        item_slots = item_slots_v1

        user_fids = inputs["user_fids"]
        user_weighted_fids = inputs["user_weighted_fids"]
        user_weighted_fid_weights = inputs["user_weighted_fid_weights"]

        fids = inputs["fids"]
        weighted_fids = inputs["weighted_fids"]
        weighted_fid_weights = inputs["weighted_fid_weights"]
        batch_size = tf.shape(fids)[0]

        user_fg = self.get_feature_embedding(
            user_fids,
            user_weighted_fids,
            user_weighted_fid_weights,
            [s.slot_id for s in user_slots],
            name="user",
            is_training=training,
            cid_dropout=False,
        )

        fg = self.get_feature_embedding(
            fids,
            weighted_fids,
            weighted_fid_weights,
            [s.slot_id for s in item_slots],
            name="item",
            is_training=training,
            cid_dropout=True,
        )

        user_embeddings = user_fg.pooled_embeddings
        item_embeddings = fg.pooled_embeddings
        if compact_predict:
            user_embeddings_bc = tf.repeat(
                user_embeddings, batch_size, axis=0, name="tile_user_embedding"
            )
        else:
            user_embeddings_bc = user_embeddings
        dnn_input = tf.concat(
            [
                tf.reshape(
                    user_embeddings_bc,
                    (-1, user_embeddings_bc.shape[1] * user_embeddings_bc.shape[2]),
                ),
                tf.reshape(
                    item_embeddings,
                    (-1, item_embeddings.shape[1] * item_embeddings.shape[2]),
                ),
            ],
            axis=1,
        )
        print("DNN INPUT", dnn_input)
        dnn_output = self.dnn(dnn_input)

        linear_input = dnn_output

        output_all = self.linear(linear_input)
        print("output all", output_all)
        if compact_predict:
            output = output_all[:, 6:7]
            output = tf.maximum(output * 3600, 600.0)
        else:
            day = inputs["day"][:, tf.newaxis]
            output = tf.gather(output_all, day, batch_dims=1)
        print("output", output)
        return {"cum_wt": output}
