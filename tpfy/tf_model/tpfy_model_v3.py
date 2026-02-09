import enum
import os
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import tensorflow_hs_addon as tfhs
from tensorflow.keras.layers import Dense
from model.layers import (
    BaseLayer,
    Linear,
    FM,
    DotProductAttentionPooling,
    DotProductAttentionPoolingV2,
    HybridDNN,
    l2_regularizer_or_none,
)

from common.config.utils import get_config
from common.config import TENANT
from tpfy.tf_dataset.feature import (
    schema,
    ScalarIndex,
    get_slot_id,
)
from model.fid import slot_bits, feature_bits

tf.compat.v1.disable_v2_behavior()

NO_USER_IDENTITY = get_config(TENANT)["general"]["NO_USER_IDENTITY"]


def get_initializer(hparams):
    if hparams.init_method == "tnormal":
        return tf.initializers.truncated_normal_initializer(stddev=hparams.init_value)
    elif hparams.init_method == "uniform":
        return tf.random_uniform_initializer(-hparams.init_value, hparams.init_value)
    elif hparams.init_method == "normal":
        return tf.random_normal_initializer(stddev=hparams.init_value)
    elif hparams.init_method == "xavier_normal":
        return tf.initializers.glorot_normal()
    elif hparams.init_method == "xavier_uniform":
        return tf.initializers.glorot_uniform()
    else:
        return tf.initializers.truncated_normal_initializer(stddev=hparams.init_value)


def find_duplication(l):
    c = Counter(l)
    for k, v in c.items():
        if v > 1:
            return k
    return None


def ensure_unique(l):
    dup = find_duplication(l)
    if dup is not None:
        raise Exception(f"Duplicate {dup}")


def _build_hash_table(values, dtype, name):
    ensure_unique(values)
    num = len(values)
    assert num > 0

    table_name = "%s_table" % name
    # index 0 is used for masking and shouldn't be used
    keys_tensor = tf.constant(values, dtype=dtype, name="%s/keys" % table_name)
    vals_tensor = tf.range(1, num + 1, dtype=tf.int32, name="%s/values" % table_name)

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        0,
        name=table_name,
    )
    return table


def _flatten_embeddings(x, name=None):
    assert len(x.shape) == 3
    dim = int(x.shape[1]) * int(x.shape[2])
    return tf.reshape(x, (-1, dim), name=name)


class FeatureGroup:
    def __init__(self):
        self.fids = None
        self.fid_slots = None
        self.fid_weights = None
        self.fid_indices = None
        self.slot_id_to_index = None
        self.unique_embeddings = None


def run_fwfm(full_user_embeddings, target_embeddings, watched_content_embeddings):
    with tf.name_scope("fwfm"):
        fm_user = tf.concat(
            [full_user_embeddings, watched_content_embeddings[:, tf.newaxis, :]], axis=1
        )
        fm_item = target_embeddings
        print("fm_user", fm_user)
        print("fm_item", fm_item)
        fwfm_output = tf.matmul(fm_item, fm_user, transpose_b=True)
        fwfm_output = tf.reshape(
            fwfm_output, [tf.shape(fm_item)[0], fm_user.shape[1] * fm_item.shape[1]]
        )
        print("fwfm out", fwfm_output)
        return fwfm_output


class TpfyModelV3(tf.keras.Model):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)

        self.drop_new_content_id_days = 2
        self.drop_new_content_id_rate = 0.1

        user_identity_slots = [
            schema.platform,
            schema.country,
            schema.state,
            schema.model,
            schema.manufacturer,
            schema.screen_height,
            schema.carrier,
            schema.asn_number,
        ]
        self.user_slots = [
            schema.gender,
            schema.age,
            schema.raw_plan_types,
            schema.joined_bucket,
            *(user_identity_slots if not NO_USER_IDENTITY else []),
            *([schema.request_country] if len(hparams.countries) > 1 else []),
            # schema.watched_content,
            schema.watched_entitlements,
            schema.watched_language_id,
            schema.watched_studio_id,
            schema.watched_genre_id,
            schema.watched_production_house_id,
            schema.watched_year_bucket,
            schema.watched_content_type,
            schema.watched_parental_rating_id,
            schema.watched_sports_team,
            schema.watched_sports_language_id,
            schema.watched_sports_tournament_id,
            schema.watched_sports_game_id,
        ]

        self.target_slots = [
            schema.target_id,
            schema.target_content_type_id,
            schema.target_genre_id,
            schema.target_studio_id,
            schema.target_production_house_id,
            schema.target_entitlement,
            schema.target_parental_rating_id,
            schema.target_year_bucket,
            schema.target_priority,
        ]
        self.slots_2d = [schema.watched_content]

        self.all_slots = self.user_slots + self.target_slots + self.slots_2d

        target_slot_ids = [s.slot_id for s in self.target_slots]
        user_slot_ids = [s.slot_id for s in self.user_slots]
        ensure_unique(target_slot_ids)
        ensure_unique(user_slot_ids)
        assert len(set(target_slot_ids) & set(user_slot_ids)) == 0

        self.sparse_feature_indices = [
            ScalarIndex.genre_weight,
            ScalarIndex.studio_weight,
            ScalarIndex.parental_rating_weight,
            ScalarIndex.content_type_weight,
            ScalarIndex.production_house_weight,
            ScalarIndex.year_bucket_weight,
            ScalarIndex.log_recency_weight,
            ScalarIndex.release_3d,
            ScalarIndex.release_7d,
            ScalarIndex.release_30d,
        ]

        self.embedding_dim = hparams.dim
        self.middle_dim = hparams.middle_dim

        self.embedding_l2 = hparams.embedding_l2
        self.dnn_l2 = hparams.dnn_l2

        self.embedding_layer = tfra.dynamic_embedding.get_variable(
            name="embedding_layer",
            key_dtype=tf.int64,
            value_dtype=tf.float32,
            dim=hparams.dim,
            initializer=get_initializer(hparams),
            devices=None,
            bp_v2=True,
        )

        # model
        self.target_attention_pooling = DotProductAttentionPoolingV2(
            dim=hparams.dim,
            activation=tf.nn.tanh,
            name="dot_prod_attention_pooling",
        )

        self.hybrid_dnn = HybridDNN(
            hidden_units=hparams.dnn_units,
            activation=hparams.dnn_activation,
            l2_reg=self.dnn_l2,
            name="dnn_hyb",
        )
        self.compress_dense = Dense(
            units=hparams.middle_dim,
            activation=None,
            kernel_initializer=tf.keras.initializers.glorot_normal(1024),
            kernel_regularizer=l2_regularizer_or_none(self.dnn_l2),
            bias_regularizer=None,
            name="compress_dense",
        )

        self.sparse_index_lookup = _build_hash_table(
            self.sparse_feature_indices, tf.int32, "sparse_index"
        )
        self.compress_activation = tf.keras.activations.relu

        self.linear_layer = Linear(
            output_units=1, l2_reg=self.dnn_l2, use_bias=True, name="linear"
        )

        self.de_trainable_wrappers = []
        self.dynamic_embeddings = [self.embedding_layer]

    def build(self, input_shape):
        self.compress_sparse_kernel = self.add_weight(
            name="sparse_layer",
            shape=(len(self.sparse_feature_indices) + 1, self.middle_dim),
            dtype=tf.float32,
            initializer=tf.initializers.zeros(),
            regularizer=l2_regularizer_or_none(self.dnn_l2),
        )
        num_task = 2
        self.task_biases = self.add_weight(
            "task_biases",
            shape=(num_task,),
            dtype=tf.float32,
            initializer=tf.initializers.zeros(),
            regularizer=None,
        )

    def query_fid(self, fids, mask_invalid=True):
        embeddings, exists = self.embedding_layer.lookup(fids, return_exists=True)
        if mask_invalid:
            mask = tf.cast(exists, tf.float32)
            embeddings = embeddings * mask[:, np.newaxis]
        return embeddings, exists

    def call(self, inputs, training=False, compact=False):
        self.log("--------------")
        batch_size = tf.shape(inputs["fids"])[0]
        task = inputs["task"]
        if training and compact:
            raise Exception("Illegal: training=True and compact=True")

        is_compact = compact
        is_predict = not training
        is_training = training
        if is_compact:
            task = tf.repeat(task, batch_size, axis=0, name="tile_task")

        with tf.name_scope("feature_prep"):
            (
                target_embeddings,
                user_embeddings,
                watched_content_embeddings,
                missed_fids,
            ) = self.get_input_embedding(
                inputs, is_compact=is_compact, is_training=is_training
            )
            self.log("target:", target_embeddings)  # [bs, nts, k]
            self.log("user:", user_embeddings)  # [1 or bs, nus, k]
            self.log("watched:", watched_content_embeddings)  # [bs, k]

            full_user_embeddings = user_embeddings
            if is_compact:
                full_user_embeddings = tf.repeat(
                    user_embeddings, batch_size, axis=0, name="tile_compact_ue"
                )

        with tf.name_scope("deepfm"):
            with tf.name_scope("dnn"):
                dnn_inputs = {
                    "full": _flatten_embeddings(
                        tf.concat(
                            [
                                target_embeddings,
                                watched_content_embeddings[:, tf.newaxis, :],
                            ],
                            axis=1,
                        ),
                        name="dnn_full_input",
                    ),
                    "compact": _flatten_embeddings(
                        user_embeddings, name="dnn_compact_input"
                    ),
                }
                dnn_output = self.hybrid_dnn(dnn_inputs)

            fm_output = run_fwfm(
                full_user_embeddings, target_embeddings, watched_content_embeddings
            )
            compress_dense_input = tf.concat([dnn_output, fm_output], axis=1)

            compress_dense_output = self.compress_dense(compress_dense_input)
            self.log("compress dense out", compress_dense_output.shape)

            # TODO: optimize index validation; costs 5% total time now
            with tf.compat.v1.variable_scope("sparse_nn"):
                sparse_indices = inputs["sparse_indices"]  # [1 or bs, num_sparse]
                sparse_values = tf.expand_dims(
                    inputs["sparse_values"], axis=1
                )  # [bs, 1, num_sparse]

                weight_indices = self.sparse_index_lookup.lookup(
                    sparse_indices, name="weight_indices"
                )

                masked_compression_sparse_kernel = tf.concat(
                    [
                        tf.zeros(
                            [1, self.compress_sparse_kernel.shape[1]], dtype=tf.float32
                        ),
                        self.compress_sparse_kernel[1:],
                    ],
                    axis=0,
                )
                print(
                    "masked_compression_sparse_kernel", masked_compression_sparse_kernel
                )
                print("compression_sparse_kernel", self.compress_sparse_kernel)
                sparse_weights = tf.gather(
                    masked_compression_sparse_kernel,
                    weight_indices,
                    name="sparse_weights",
                )  # [1 or bs, num_sparse, out]
                compress_sparse_output = tf.matmul(
                    sparse_values, sparse_weights, name="sparse_matmul"
                )  # [bs, 1, out]
                compress_sparse_output = tf.squeeze(compress_sparse_output, axis=1)
                self.log("compress sparse out", compress_sparse_output)

            compress_output = compress_dense_output + compress_sparse_output
            compress_output = self.compress_activation(compress_output)
            self.log("compress_output", compress_output)

            linear_input = compress_output
            target_output = self.linear_layer(linear_input)
            self.log("target_output shape", target_output.shape)
            self.log("bias shape", self.task_biases[0].shape)

            clickwatch = target_output + tf.reshape(self.task_biases[0], (1, 1))
            lastwatch = target_output + tf.reshape(self.task_biases[1], (1, 1))

            if compact:
                clickwatch = tf.sigmoid(clickwatch)
                lastwatch = tf.sigmoid(lastwatch)

            task_biases = tf.gather(self.task_biases, task)
            self.log("task bias shape", task_biases.shape)

            uni_output = target_output + task_biases
            print("model output", uni_output, clickwatch, lastwatch)

            tf.summary.histogram("Predictor/output", uni_output)
            return {
                "scores": uni_output,
                "missed_fids": missed_fids,
                "clickwatch": clickwatch,
                "lastwatch": lastwatch,
            }

    def log(self, *args, **kwargs):
        print(*args, **kwargs)

    def get_input_embedding(self, mini_batch, is_compact, is_training):
        # dropout new fids
        fids = mini_batch["fids"]
        if (
            is_training
            and self.drop_new_content_id_days > 0
            and self.drop_new_content_id_rate > 0
        ):
            secs_start_dt = mini_batch["secs_start_dt"]
            ex_should_drop = tf.logical_and(
                secs_start_dt <= 86400 * self.drop_new_content_id_days,
                tf.random.uniform(tf.shape(secs_start_dt))
                < self.drop_new_content_id_rate,
            )
            fid_slots = tf.bitwise.right_shift(fids, feature_bits)
            fid_should_drop = tf.logical_and(
                tf.equal(fid_slots, schema.target_id.slot_id),
                ex_should_drop[:, tf.newaxis],
            )
            fids = tf.where(fid_should_drop, tf.constant(0, dtype=tf.int64), fids)

        target_fg, target_embeddings, target_missed_fids = self.get_feature_embedding(
            fids,
            mini_batch["weighted_fids"],
            mini_batch["weighted_fid_weights"],
            [s.slot_id for s in self.target_slots],
            name="target",
            is_training=is_training,
            additional_slots_to_query=[],
        )
        user_fg, user_embeddings, user_missed_fids = self.get_feature_embedding(
            mini_batch["user_fids"],
            mini_batch["user_weighted_fids"],
            mini_batch["user_weighted_fid_weights"],
            [s.slot_id for s in self.user_slots],
            name="user",
            is_training=is_training,
            additional_slots_to_query=[s.slot_id for s in self.slots_2d],
        )

        target_id_slot_index = target_fg.slot_id_to_index[schema.target_id.slot_id]
        target_id_embedding = target_embeddings[:, target_id_slot_index, :]
        watched_content_fid_indices, watched_content_fid_weights = tfhs.get_slot_fids(
            user_fg.fid_indices,
            user_fg.fid_slots,
            user_fg.fid_weights,
            schema.watched_content.slot_id,
        )

        watched_content_embedding_unpooled = tf.gather(
            user_fg.unique_embeddings,
            watched_content_fid_indices,
            name="watched_content_embedding_unpooled",
        )

        watched_content_embedding = self.target_attention_pooling(
            [
                target_id_embedding,
                watched_content_embedding_unpooled,
                watched_content_fid_weights,
            ],
            training=is_training,
        )

        if not is_training:
            missed_fids = tf.concat([user_missed_fids, target_missed_fids], axis=0)
        else:
            missed_fids = None

        print("target embedding shape", target_embeddings.shape)
        print("user embedding shape", user_embeddings.shape)
        return (
            target_embeddings,
            user_embeddings,
            watched_content_embedding,
            missed_fids,
        )

    def get_feature_embedding(
        self,
        fids,
        weighted_fids,
        weighted_fid_weights,
        slots_to_pool,
        name,
        is_training,
        additional_slots_to_query=[],
    ):
        with tf.name_scope(f"{name}_feature"):
            fid_weights = tf.concat(
                [tf.ones_like(fids, dtype=tf.float32), weighted_fid_weights],
                axis=1,
                name="fid_weight_concat",
            )
            fids = tf.concat([fids, weighted_fids], axis=1, name="fid_concat")

            fid_slots = tf.cast(
                tf.bitwise.right_shift(fids, feature_bits), tf.int32, name="fid_slots"
            )
            fids = tf.where(
                tfhs.isin(fid_slots, slots_to_pool + additional_slots_to_query), fids, 0
            )

            flat_fids = tf.reshape(fids, (-1,))
            unique_fids, flat_indices = tf.unique(flat_fids, name="fid_dedup")
            fid_indices = tf.reshape(flat_indices, tf.shape(fids), name="fid_indices")

            if is_training:
                unique_embeddings, wrapper = tfra.dynamic_embedding.embedding_lookup(
                    self.embedding_layer,
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
                unique_embeddings, exists = self.embedding_layer.lookup(
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
            return fg, pooled_embeddings, missed_fids

    def get_all_trainable_variables(self):
        return self.trainable_variables + self.de_trainable_wrappers

    def get_all_savable_variables(self):
        return [
            v
            for v in self.variables
            if not isinstance(v, tfra.dynamic_embedding.TrainableWrapper)
        ]

    # used to export model weights from distributed version to local
    def export_plain_weights_ops(self, optimizer=None):
        if not self.built:
            raise Exception("Model not built")
        savable_variables = self.get_all_savable_variables()
        export_ops = {}
        for v in savable_variables:
            export_ops[v.name] = v
        for de in self.dynamic_embeddings:
            fids, embeddings = de.export()
            export_ops[f"{de.name}/fids"] = fids
            export_ops[f"{de.name}/embeddings"] = embeddings

        if optimizer is not None:
            for de in self.dynamic_embeddings:
                for de_slot_variable in de.get_slot_variables(optimizer):
                    opt_slot_fids, opt_slot_weights = de_slot_variable.export()
                    export_ops[f"{de_slot_variable.name}/fids"] = opt_slot_fids
                    export_ops[f"{de_slot_variable.name}/weights"] = opt_slot_weights

        return export_ops

    def restore_plain_weights_ops(self, plain_weights, optimizer=None, clear_nn=False):
        savable_variables = self.get_all_savable_variables()
        restore_ops = []

        print("reload source keys")
        for k in plain_weights:
            print(k)
        if not clear_nn:
            print("resolve dense parameters")
            for v in savable_variables:
                print("resolving", v.name)
                w = None
                if v.name in plain_weights:
                    w = plain_weights[v.name]
                    print("hit", v.name)
                elif v.name.startswith("train/tpfy_model_v3/"):
                    legacy_name = v.name.replace("train/tpfy_model_v3/", "tpfy_model/")
                    if legacy_name in plain_weights:
                        print("hit legacy", legacy_name)
                        w = plain_weights[legacy_name]
                if w is None:
                    raise Exception(f"weights of {v.name} doesn't exist")
                if v.shape != w.shape:
                    raise Exception(f"{v.name} shape mismatch; skip", v.shape, w.shape)
                print("restore savable variable", v.name, v)
                restore_ops.append(tf.compat.v1.assign(v, w))
        for de in self.dynamic_embeddings:
            print("resolving embedding", de.name)
            fid_name = f"{de.name}/fids"
            embedding_name = f"{de.name}/embeddings"
            if (fid_name not in plain_weights) or (embedding_name not in plain_weights):
                raise Exception(f"weights of de {de.name} doesn't exist")
            fids = plain_weights[fid_name]
            embeddings = plain_weights[embedding_name]
            restore_ops.append(de.upsert(fids, embeddings))

        if optimizer is not None:
            for de in self.dynamic_embeddings:
                for de_slot_variable in de.get_slot_variables(optimizer):
                    opt_slot_fids_name = f"{de_slot_variable.name}/fids"
                    opt_slot_weights_name = f"{de_slot_variable.name}/weights"
                    if (opt_slot_fids_name not in plain_weights) or (
                        opt_slot_weights_name not in plain_weights
                    ):
                        raise Exception(
                            f"opt slot weights of de {de.name} doesn't exist; "
                            f"slot variable names: {opt_slot_fids_name}, {opt_slot_weights_name}"
                        )
                    opt_slot_fids = plain_weights[opt_slot_fids_name]
                    opt_slot_weights = plain_weights[opt_slot_weights_name]
                    restore_ops.append(
                        de_slot_variable.upsert(opt_slot_fids, opt_slot_weights)
                    )

        return restore_ops
