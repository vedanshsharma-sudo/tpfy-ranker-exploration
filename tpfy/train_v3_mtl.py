import argparse
import json
import time
from dataclasses import dataclass

import tensorflow as tf

tfv1 = tf.compat.v1
tfv1.disable_v2_behavior()

import os
import pyarrow

from common.time_utils import get_dates_list_forwards
import tensorflow_addons as tfa
import tensorflow_recommenders_addons as tfra

import numpy as np
import s3fs
from common.s3_utils import upload_folder, upload_file

from common.config.utils import data_path, tenant_countries, model_path
from common.config import TENANT

from common.s3_utils import is_s3_path_success
from model.trainer import Trainer, ValData, LearningRateScheduler
from model.losses import masked_binary_entropy_loss
from model.metrics import MaskedAUC
from tpfy.tf_model.tpfy_model_v3_mtl import TpfyModelV3, TpfyMtlModelConfig
from tpfy.common import TpfyDataPath
import tpfy.tf_model.exporter_v3_mtl as exporter
from tpfy.etl.schema import TpfyMtlDatasetSchema
from model.parquet_dataset import TFParquetDataset, _get_dataset_columns
from omegaconf import OmegaConf


S3_TPFY_MODEL_EXPORT = model_path(TpfyDataPath.S3_TPFY_MODEL_EXPORT, TENANT)


@dataclass
class TpfyTrainConfig:
    repeat: int
    eval_freq: int
    eval_steps: int
    step_unit: int
    max_step: int
    batch_size: int

    learning_rate: float
    lr_decay: bool
    lr_decay_start: int
    min_lr: float
    weight_decay: float

    enable_random_watch: bool


@dataclass
class TpfyConfig:
    train: TpfyTrainConfig
    model: TpfyMtlModelConfig


_dataset_column_names = [col.name for col in _get_dataset_columns(TpfyMtlDatasetSchema)]

TASK_COL_INDEX = _dataset_column_names.index("task")


def filter_random_watch_factory():
    def _fn(imm_row):
        task = imm_row[TASK_COL_INDEX]
        if task[0] == 0:
            return [imm_row]
        else:
            return []

    return _fn


def init_local(args, countries) -> TpfyConfig:
    name = f"tpfy/tpfy_config/mtl-{TENANT}.yaml"

    if not os.path.exists(name):
        raise Exception(f"conf file {name} missing")

    if args.conf:
        cli_conf_dotlist = [p for p in args.conf.split(",") if len(p) > 0]
    else:
        cli_conf_dotlist = []

    conf: TpfyConfig = OmegaConf.merge(
        OmegaConf.structured(TpfyConfig),
        OmegaConf.load(name),
        OmegaConf.from_dotlist(cli_conf_dotlist),
    )

    if args.lr:
        conf.train.learning_rate = args.lr

    if args.eval_freq is not None:
        conf.train.eval_freq = args.eval_freq

    if args.batch_size:
        conf.train.batch_size = args.batch_size

    if args.max_epoch:
        conf.train.max_step = args.max_epoch

    if args.repeat:
        conf.train.repeat = args.repeat

    return conf


def assert_label_shape(tensor):
    assert len(tensor.shape) == 2
    assert tensor.shape[1].value == 1


def partition_columns(values):
    schema = TpfyMtlDatasetSchema
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


def make_example_mtl(*tensors):
    features, original_labels, metadata = partition_columns(tensors)

    task = features["task"]
    click = tf.cast(original_labels.click, tf.float32)
    watch = tf.cast(original_labels.watch, tf.float32)
    paywall_view = tf.cast(original_labels.paywall_view, tf.float32)
    add_watchlist = tf.cast(original_labels.add_watchlist, tf.float32)

    click = tf.where(tf.equal(task, 0), click, -1.0)
    random_watch = tf.where(tf.equal(task, 1), watch, -1.0)

    is_postclick = tf.greater(click, 0)
    watch = tf.where(is_postclick, watch, -1.0)
    paywall_view = tf.where(is_postclick, paywall_view, -1.0)
    add_watchlist = tf.where(is_postclick, add_watchlist, -1.0)

    assert_label_shape(click)
    assert_label_shape(watch)
    assert_label_shape(random_watch)

    labels = {
        "click": click,
        "watch": watch,
        "add_watchlist": add_watchlist,
        "paywall_view": paywall_view,
        "random_watch": random_watch,
    }

    return features, labels, metadata


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
    print(hparams)

    variant = args.variant
    if variant and not variant.startswith("-"):
        variant = "-" + variant

    train_date = args.date
    train_path = data_path(
        TpfyDataPath.S3_TPFY_IMPR_V3_AGG_MTL_EXTRACTED_EXAMPLES_VAR, TENANT
    ) % (variant, train_date)
    print("train data", train_path)
    if not is_s3_path_success(train_path):
        raise Exception("train data not available")
    dataset_schema = TpfyMtlDatasetSchema

    train_dataset = TFParquetDataset([train_path], dataset_schema, shuffle_files=True)
    train_row_transformer_factory = None

    make_example_fn = make_example_mtl

    session = tfv1.keras.backend.get_session()

    print("load dataset objective stat")
    fs = s3fs.S3FileSystem(use_ssl=False)
    with fs.open(os.path.join(train_path, "stats.json"), "r") as f:
        stats = json.load(f)
    task_weights = stats["task_weights"]

    batch_size = hparams.train.batch_size
    print(f"obj stat: batch_size {batch_size}")

    train_tf_dataset = train_dataset.create_parallel_tf_dataset(
        batch_size,
        args.num_workers,
        num_epochs=hparams.train.repeat,
        queue_size=16,
        v2=True,
        row_transformer_factory=train_row_transformer_factory,
    ).map(make_example_fn)

    print("output_shapes", train_tf_dataset.output_shapes)
    print("num output", len(train_tf_dataset.output_shapes))

    validation_dataset_dict = {}
    val_tenant_or_countries = [TENANT]
    for country in val_tenant_or_countries:
        for dt in get_dates_list_forwards(args.val_date, args.val_days):
            val_dataset = TFParquetDataset(
                [
                    data_path(
                        TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_MTL_EXTRACTED_EXAMPLES,
                        country,
                    )
                    % (variant, dt)
                ],
                dataset_schema,
                shuffle_files=False,
            )
            validation_dataset_dict[f"{country}-{dt}"] = ValData(
                val_dataset.create_tf_dataset(batch_size)
                .take(hparams.train.eval_steps)
                .cache(f"val_mtl_{country}_{dt}")
                .map(make_example_fn),
                active_objectives=[
                    "click",
                    "watch",
                    "random_watch",
                    "paywall_view",
                    "add_watchlist",
                ],
            )

    model_name = args.model_name
    tpfy_model = TpfyModelV3(
        hparams.model,
        click_ns=args.click_ns,
        enable_random_watch=hparams.train.enable_random_watch,
    )

    optimizer = tfa.optimizers.AdamW(
        weight_decay=float(hparams.train.weight_decay),
        learning_rate=hparams.train.learning_rate,
        epsilon=1e-4,
    )
    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

    if hparams.train.lr_decay:
        lr_scheduler = create_exp_lr_schedule_callback(
            hparams.train.lr_decay_start,
            hparams.train.min_lr,
            0.7 ** (1 / 10),
            verbose=True,
        )
    else:
        lr_scheduler = None

    loss_dict = {
        "click": masked_binary_entropy_loss(from_logits=True),
        "watch": masked_binary_entropy_loss(from_logits=True),
        "random_watch": masked_binary_entropy_loss(from_logits=False),
        "paywall_view": masked_binary_entropy_loss(from_logits=True),
        "add_watchlist": masked_binary_entropy_loss(from_logits=True),
    }
    metric_dict = {
        "click": MaskedAUC(from_logits=True),
        "watch": MaskedAUC(from_logits=True),
        "random_watch": MaskedAUC(from_logits=False),
        "paywall_view": MaskedAUC(from_logits=True),
        "add_watchlist": MaskedAUC(from_logits=True),
    }

    loss_weight_dict = {
        "click": args.click_weight,
        "watch": args.watch_weight,
        "random_watch": 1.0 if hparams.train.enable_random_watch else 0.0,
        "paywall_view": 1.0,
        "add_watchlist": 1.0,
    }
    total_loss_weight = sum(loss_weight_dict.values())
    loss_weight_dict = {
        obj: w / total_loss_weight for obj, w in loss_weight_dict.items()
    }

    tpfy_model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        metrics=metric_dict,
        loss_weights=loss_weight_dict,
    )
    plain_weights = None
    clear_nn = args.clear_nn
    if args.reload_s3_model or args.reload_local_model:
        if args.reload_s3_model:
            filesystem = s3fs.S3FileSystem(use_ssl=False)
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
        epochs=hparams.train.max_step,
        steps_per_epoch=hparams.train.step_unit,
        validation_data_dict=validation_dataset_dict,
        validation_steps=None,
        validation_freq=hparams.train.eval_freq,
        log_dir="train/logs/" + model_name,
        lr_scheduler=lr_scheduler,
        early_stopping=None,
        verbose=args.verbose,
        validation_on_start=hparams.train.eval_freq > 0
        and (args.reload_local_model or args.reload_s3_model),
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
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--val_days", type=int, default=1)
    parser.add_argument("--click_ns", type=float)
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--click_weight", type=float, default=1)
    parser.add_argument("--watch_weight", type=float, default=1)
    parser.add_argument("--upload", action="store_true", help="uploading model to s3")
    parser.add_argument("--reload_local_model", type=str, default=None)
    parser.add_argument("--reload_s3_model", type=str, default=None)
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
