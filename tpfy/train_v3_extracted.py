import argparse
import json
import time

import tensorflow as tf

tfv1 = tf.compat.v1
tfv1.disable_v2_behavior()

import os
import pyarrow
from collections import defaultdict

from common.time_utils import get_dates_list_backwards, get_dates_list_forwards
import tensorflow_addons as tfa
import tensorflow_recommenders_addons as tfra

import numpy as np
import s3fs
from common.s3_utils import upload_folder, upload_file

from common.config.constants import DataPath
from common.config.utils import data_path, tenant_countries, model_path
from common.config import TENANT

from tpfy.tpfy_config import get_tpfy_hparams
from common.s3_utils import is_s3_path_success
from model.trainer import Trainer, ValData, LearningRateScheduler
from model.losses import masked_binary_entropy_loss, statnorm_masked_binary_entropy_loss
from model.metrics import MaskedAUC
from tpfy.tf_model.tpfy_model_v3 import TpfyModelV3
from tpfy.common import TpfyDataPath
import tpfy.tf_model.exporter_v3 as exporter
from tpfy.etl.schema import TpfyDatasetSchema
from model.parquet_dataset import TFParquetDataset, _get_dataset_columns


LEGACY_S3_TPFY_MODEL_EXPORT = data_path(DataPath.S3_TPFY_MODEL_EXPORT, TENANT, "")

S3_TPFY_MODEL_EXPORT = model_path(TpfyDataPath.S3_TPFY_MODEL_EXPORT, TENANT)


def init_local(args, countries):
    name = f"network-{TENANT}.yaml"
    hparams = get_tpfy_hparams(fname=name)
    return hparams


def assert_label_shape(tensor):
    assert len(tensor.shape) == 2
    assert tensor.shape[1].value == 1


def partition_columns(values):
    schema = TpfyDatasetSchema
    num_features = len(schema.features)
    num_labels = len(schema.labels)
    num_metadata = len(schema.metadata)
    offset = 0
    features = schema.make_feature_tuple(
        values[offset : offset + num_features]
    )._asdict()
    offset += num_features

    labels = schema.make_label_tuple(values[offset : offset + num_labels])
    offset += num_labels

    metadata = schema.make_metadata_tuple(
        values[offset : offset + num_metadata]
    )._asdict()
    return features, labels, metadata


def make_example_click(*values):
    features, labels, metadata = partition_columns(values)

    click = tf.cast(labels.click, tf.float32)
    labels = {"click": click}
    return features, labels, metadata


def make_example_sep(*tensors):
    features, original_labels, metadata = partition_columns(tensors)

    task = features["task"]
    label = tf.cast(original_labels.label, tf.float32)

    clickwatch = tf.where(tf.equal(task, 0), label, -1.0)
    lastwatch = tf.where(tf.equal(task, 1), label, -1.0)

    assert_label_shape(label)
    assert_label_shape(clickwatch)
    assert_label_shape(lastwatch)

    labels = {
        "scores": label,
        "clickwatch": clickwatch,
        "lastwatch": lastwatch,
    }

    return features, labels, metadata


_dataset_column_names = [col.name for col in _get_dataset_columns(TpfyDatasetSchema)]

TASK_COL_INDEX = _dataset_column_names.index("task")


def clickwatch_row_filter_facotry():
    def _fn(imm_row):
        task = imm_row[TASK_COL_INDEX]
        if task[0] == 0:
            return [imm_row]
        else:
            return []

    return _fn


def lastwatch_row_filter_facotry():
    def _fn(imm_row):
        task = imm_row[TASK_COL_INDEX]
        if task[0] == 1:
            return [imm_row]
        else:
            return []

    return _fn


def create_exp_lr_schedule_callback(decay_start, min_lr, alpha, verbose=False):
    def lr_schedule(epoch, current_lr):
        if epoch < decay_start:
            return current_lr
        else:
            lr = current_lr * alpha
            if lr < min_lr:
                return min_lr
            else:
                return lr

    return LearningRateScheduler(lr_schedule, verbose=verbose)


def create_multistage_lr_schedule_callback(stages, verbose=False):
    def lr_schedule(epoch, current_lr):
        target = None
        for i, stage in enumerate(stages):
            if epoch < stage[0]:
                target = i - 1
                break
        if target is None:
            target = len(stages) - 1
        if target < 0:
            return current_lr
        else:
            return stages[target][1]

    return LearningRateScheduler(lr_schedule, verbose=verbose)


class TpfyCustomTrainer(Trainer):
    def __init__(
        self,
        model: TpfyModelV3,
        session,
        model_name,
        plain_weights,
        clear_nn,
        weight_decay,
        countries,
    ):
        super().__init__(model=model, model_name=model_name, session=session)
        self.plain_weights = plain_weights
        self.clear_nn = clear_nn
        self.weight_decay = weight_decay
        self.countries = countries

        assert isinstance(model, TpfyModelV3)

    def on_train_start(self):
        print("on train start")
        if self.plain_weights is not None:
            print("restore model weights")

            plain_weights = self.plain_weights
            restore_ops = self.model.restore_plain_weights_ops(
                plain_weights, clear_nn=self.clear_nn
            )
            self.session.run(restore_ops)
            print("done")

    def build_train_step(self, tape, loss):
        if not self.weight_decay:
            trainable_vars = self.model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            train_step = self.model.optimizer.apply_gradients(
                zip(gradients, trainable_vars)
            )
            return train_step
        else:
            print("build weight decay")
            trainable_variables = self.model.get_all_trainable_variables()
            print("trainable", [v.name for v in trainable_variables])
            decayed_variables = [v for v in trainable_variables if "bias" not in v.name]

            train_step = self.model.optimizer.minimize(
                loss,
                trainable_variables,
                decay_var_list=decayed_variables,
                tape=tape,
            )
            return train_step


def run(args):
    countries = tenant_countries(args.countries)
    hparams = init_local(args, countries)
    hparams.countries = countries

    if args.lr:
        hparams.learning_rate = args.lr

    if args.eval_freq is not None:
        hparams.eval_freq = args.eval_freq

    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = hparams.batch_size

    if args.max_epoch:
        hparams.max_step = args.max_epoch

    variant = args.variant
    if variant and not variant.startswith("-"):
        variant = "-" + variant

    train_date = args.date
    train_path = data_path(
        TpfyDataPath.S3_TPFY_IMPR_V3_AGG_EXTRACTED_EXAMPLES_VAR, TENANT
    ) % (variant, train_date)
    print("train data", train_path)
    if not is_s3_path_success(train_path):
        raise Exception("train data not available")
    dataset_schema = TpfyDatasetSchema

    train_dataset = TFParquetDataset([train_path], dataset_schema, shuffle_files=True)
    train_row_transformer_factory = None

    make_example_fn = make_example_sep

    session = tfv1.keras.backend.get_session()

    if args.statnorm:
        print("load dataset objective stat")
        fs = s3fs.S3FileSystem(use_ssl=False)
        with fs.open(os.path.join(train_path, "stats.json"), "r") as f:
            stats = json.load(f)
        task_weights = stats["task_weights"]

        num_obj_active = {
            "clickwatch": task_weights["0"] * batch_size,
            "lastwatch": task_weights["1"] * batch_size,
        }
        print("obj weights", num_obj_active)
        print(f"obj stat: batch_size {batch_size}")

    train_tf_dataset = train_dataset.create_parallel_tf_dataset(
        batch_size,
        args.num_workers,
        num_epochs=args.repeat,
        queue_size=16,
        v2=True,
        row_transformer_factory=train_row_transformer_factory,
    ).map(make_example_fn)

    print("output_shapes", train_tf_dataset.output_shapes)
    print("num output", len(train_tf_dataset.output_shapes))

    validation_dataset_dict = {}
    if args.val_tenantwise:
        val_tenant_or_countries = [TENANT]
    else:
        val_tenant_or_countries = tenant_countries()
    for country in val_tenant_or_countries:
        for dt in get_dates_list_forwards(args.val_date, args.val_days):
            val_dataset = TFParquetDataset(
                [
                    data_path(
                        TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_EXTRACTED_EXAMPLES, country
                    )
                    % (variant, dt)
                ],
                dataset_schema,
                shuffle_files=False,
            )
            validation_dataset_dict[f"{country}-clickwatch-{dt}"] = ValData(
                val_dataset.create_tf_dataset(
                    batch_size, row_transformer_factory=clickwatch_row_filter_facotry
                )
                .take(hparams.eval_count_impr)
                .cache(f"val_clickwatch_{country}_{dt}")
                .map(make_example_fn),
                active_objectives=["scores"],
            )
            validation_dataset_dict[f"{country}-lastwatch-{dt}"] = ValData(
                val_dataset.create_tf_dataset(
                    batch_size, row_transformer_factory=lastwatch_row_filter_facotry
                )
                .take(hparams.eval_count)
                .cache(f"val_lastwatch_{country}_{dt}")
                .map(make_example_fn),
                active_objectives=["scores"],
            )

    model_name = args.model_name
    tpfy_model = TpfyModelV3(hparams)

    optimizer = tfa.optimizers.AdamW(
        weight_decay=float(hparams.weight_decay),
        learning_rate=hparams.learning_rate,
        epsilon=1e-3,
    )
    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

    # lr_scheduler = create_exp_lr_schedule_callback(
    #     40, 0.0001, 0.7 ** (1 / 10), verbose=True
    # )
    lr_scheduler = None

    if not args.sepobj:
        loss_dict = {"scores": tf.losses.BinaryCrossentropy(from_logits=True)}
        metric_dict = {
            "scores": tf.metrics.AUC(from_logits=True),
            "clickwatch": MaskedAUC(from_logits=True),
            "lastwatch": MaskedAUC(from_logits=True),
        }
        loss_weight_dict = {"scores": 1.0}
        print("UNI obj")
    else:
        if args.statnorm:
            loss_dict = {
                "clickwatch": statnorm_masked_binary_entropy_loss(
                    num_obj_active["clickwatch"], from_logits=True
                ),
                "lastwatch": statnorm_masked_binary_entropy_loss(
                    num_obj_active["lastwatch"], from_logits=True
                ),
            }
        else:
            loss_dict = {
                "clickwatch": masked_binary_entropy_loss(from_logits=True),
                "lastwatch": masked_binary_entropy_loss(from_logits=True),
            }
        metric_dict = {
            "scores": tf.metrics.AUC(from_logits=True),
            "clickwatch": MaskedAUC(from_logits=True),
            "lastwatch": MaskedAUC(from_logits=True),
        }
        loss_weight_dict = {"clickwatch": args.clickwatch_loss_weight, "lastwatch": 1.0}
        total_loss_weight = sum(loss_weight_dict.values())
        loss_weight_dict = {
            obj: w / total_loss_weight for obj, w in loss_weight_dict.items()
        }
        print("SEP obj")

    tpfy_model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        metrics=metric_dict,
        loss_weights=loss_weight_dict,
    )

    """
    it = train_dataset.make_one_shot_iterator()
    nxt = it.get_next()
    print("TEST")
    total = 0
    random_total = 0
    ratio_list = []
    for i in range(1000):
        if i % 100 == 1:
            print(i)
        f, l, m = session.run(nxt)
        lbl = l['scores'].reshape(-1)
        task = f['task'].reshape(-1)
        cw = l['clickwatch'].reshape(-1)
        lw = l["lastwatch"].reshape(-1)
        bs = len(task)
        num_random = task.sum()
        total += bs
        random_total += num_random
        ratio_list.append(num_random / bs)
        # print("task", task.sum(), task[:20])
        # print("label", lbl.sum(), lbl[:20])
        # print("clickwatch", (cw>0).sum(), cw[:20])
        # print("lastwatch", (lw>0).sum(), lw[:20])
    print(f"total {total}, random {random_total}, rel_weight: {(total-random_total) / random_total}")
    print("ratio:", np.min(ratio_list), np.max(ratio_list), np.mean(ratio_list), np.std(ratio_list))
    return
    """

    plain_weights = None
    clear_nn = args.clear_nn
    if args.reload_s3_model or args.reload_local_model:
        if args.reload_s3_model:
            filesystem = s3fs.S3FileSystem(use_ssl=False)
            if args.reload_s3_legacy:
                model_path = LEGACY_S3_TPFY_MODEL_EXPORT % args.reload_s3_model
            else:
                model_path = S3_TPFY_MODEL_EXPORT % args.reload_s3_model
        else:
            filesystem = pyarrow.LocalFileSystem()
            model_path = os.path.join("export", args.reload_local_model)

        if args.ckpt:
            checkpoint = args.ckpt
        else:
            checkpoint_path = os.path.join(model_path, "checkpoint")
            print("read checkpoint", checkpoint_path)
            with filesystem.open(checkpoint_path, "r") as f:
                checkpoint = f.read().strip()
        weights_path = os.path.join(model_path, checkpoint, "plain_weights.npz")
        if not filesystem.exists(weights_path):
            raise Exception(f"Model weights {weights_path} unavailable")
        else:
            print(f"Restore from {weights_path}")
            with filesystem.open(weights_path, "rb") as f:
                plain_weights = {}
                for k, v in np.load(f).items():
                    plain_weights[k] = v

            print("plain weights keys", list(plain_weights.keys()))

    trainer = TpfyCustomTrainer(
        tpfy_model,
        session,
        model_name,
        plain_weights,
        clear_nn=clear_nn,
        weight_decay=True,
        countries=countries,
    )
    trained_epochs = trainer.train(
        train_tf_dataset,
        epochs=hparams.max_step,
        steps_per_epoch=hparams.step_unit,
        validation_data_dict=validation_dataset_dict,
        validation_steps=None,
        validation_freq=hparams.eval_freq,
        log_dir="train/logs/" + model_name,
        lr_scheduler=lr_scheduler,
        early_stopping=None,
        verbose=args.verbose,
        validation_on_start=hparams.eval_freq > 0,
    )

    # WORKAROUND: clear de optimizer state to save memory online
    # TODO: remove optimizer entirely
    for de_var in tpfy_model.dynamic_embeddings:
        print("clear optimizer state for embedding variable", de_var.name)
        for i, opt in enumerate([optimizer]):
            print("opt", i)
            slot_variables = de_var.get_slot_variables(opt)
            for slot_variable in slot_variables:
                print(
                    "clear",
                    slot_variable.name,
                    "size",
                    session.run(slot_variable.size()),
                )
                session.run(slot_variable.clear())

    warmup_dataset = list(validation_dataset_dict.values())[0].tf_dataset
    warmup_it = warmup_dataset.make_one_shot_iterator()
    warmup_next = warmup_it.get_next()
    warmup_data = []
    print("get warmup data")
    for i in range(1):
        warmup_data.append(session.run(warmup_next))
    print("warmup data done")

    model_dir = "export/{}".format(model_name)
    version = int(time.time())
    export_path = "{}/{}".format(model_dir, version)
    print("export path", export_path)
    # from IPython import embed
    # embed()
    exporter.export_tpfy_model(export_path, session, tpfy_model, warmup_data)
    checkpoint_path = os.path.join(model_dir, "checkpoint")
    with open(checkpoint_path, "w") as f:
        f.write(str(version))

    if args.upload:
        print("upload")
        export_s3_path = S3_TPFY_MODEL_EXPORT % f"{model_name}/{version}"
        upload_folder(export_s3_path, export_path, set_acl=False)

        checkpoint_s3_path = S3_TPFY_MODEL_EXPORT % f"{model_name}/checkpoint"
        upload_file(checkpoint_s3_path, checkpoint_path, set_acl=False)

    print("done")


def main():
    parser = argparse.ArgumentParser(description="TPFY offline Training.")
    parser.add_argument("model_name", type=str)
    parser.add_argument("date", type=str)
    parser.add_argument("val_date", type=str)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--val_tenantwise", action="store_true")
    parser.add_argument("--val_days", type=int, default=1)
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sepobj", action="store_true")
    parser.add_argument("--statnorm", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--all_lw", action="store_true")
    parser.add_argument("--no_impr_random_neg", action="store_true")
    parser.add_argument("--clickwatch_loss_weight", type=float, default=1.0)
    parser.add_argument("--upload", action="store_true", help="uploading model to s3")
    parser.add_argument("--enable_metrics", action="store_true")
    parser.add_argument("--reload_local_model", type=str, default=None)
    parser.add_argument("--reload_s3_model", type=str, default=None)
    parser.add_argument("--reload_s3_legacy", action="store_true")
    parser.add_argument("--clear_nn", action="store_true")
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--countries",
        type=str,
        help="countries to run, separated with comma. "
        "default is None. fallback to region countries",
    )
    args = parser.parse_args()
    print("Start training")
    run(args)


if __name__ == "__main__":
    main()
