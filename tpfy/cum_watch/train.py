import argparse
import json
import time
import os
import numpy as np
from common.config import TENANT

import tensorflow as tf

tfv1 = tf.compat.v1
tfv1.disable_v2_behavior()

from common.time_utils import get_dates_list_backwards, get_dates_list_forwards
from common.s3_utils import upload_folder, upload_file
from common.config.utils import tenant_countries, data_path, model_path
from tpfy.cum_watch.cum_watch_schema import CumWatchDataSchema
from tpfy.cum_watch.cum_watch_model import CumWatchModel
from tpfy.common import TpfyDataPath
from model.parquet_dataset import TFParquetDataset
from model.trainer import Trainer, LearningRateScheduler, EarlyStopping
import tensorflow_recommenders_addons as tfra
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.signature_constants import PREDICT_METHOD_NAME
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def
from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.saved_model.utils import build_tensor_info

from tensorflow_serving.apis import predict_pb2, model_pb2, prediction_log_pb2

schema = CumWatchDataSchema


def partition_columns(values):
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


def make_cum_wt_example(*values):
    features, labels, metadata = partition_columns(values)

    cum_watch_time = tf.cast(labels.cum_watch_time, tf.float32)
    labels = {"cum_wt": cum_watch_time}
    return features, labels, metadata


def make_all_wt_example(*values):
    features, labels, metadata = partition_columns(values)

    cum_watch_time = tf.cast(labels.all_watch_time, tf.float32)
    labels = {"cum_wt": cum_watch_time}
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


def build_signature(
    inputs,
    outputs,
):
    input_dict = {}
    for name, placeholder in inputs.items():
        input_dict[name] = build_tensor_info(placeholder)
    output_dict = {}
    for name, output_tensor in outputs.items():
        output_dict[name] = build_tensor_info(output_tensor)
    predict_signature = build_signature_def(
        inputs=input_dict,
        outputs=output_dict,
        method_name=PREDICT_METHOD_NAME,
    )
    return predict_signature


def run(args):
    evaluate_date = args.eval_date
    num_workers = args.num_workers
    model_name = args.model_name

    print("eval:", evaluate_date)

    if args.train_days == 0:
        raise Exception("train_days is 0")

    train_variant = args.train_variant
    if train_variant is None:
        train_variant = ""
    elif len(train_variant) > 0 and not train_variant.startswith("-"):
        train_variant = "-" + train_variant
    train_dates = get_dates_list_backwards(args.train_date, args.train_days)
    print("train:", ",".join(train_dates))

    train_countries = tenant_countries(args.countries)

    # TODO: remove hack
    train_countries = list(set(train_countries) - {"ph"})

    train_paths = []
    for country in train_countries:
        train_paths.extend(
            [
                data_path(TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_EXAMPLE_D7, country)
                % (train_variant, date)
                for date in train_dates
            ]
        )
        train_paths.append(
            data_path(TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_EXAMPLE_PROG, country)
            % (train_variant, args.train_date)
        )
    shuffle = True

    print("create model")
    optimizer = tf.optimizers.Adam(learning_rate=args.lr)
    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

    use_featuredump = train_variant != ""
    model = CumWatchModel(dim=16, use_featuredump=use_featuredump)
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer=optimizer,
        loss={"cum_wt": loss},
        metrics={
            "cum_wt": [
                tf.keras.metrics.RootMeanSquaredError(),
            ]
        },
    )
    validation_steps = 1000
    batch_size = 256
    steps_per_epoch = 1000
    train_max_epochs = args.max_epoch

    if args.all_wt:
        make_ex_fn = make_all_wt_example
    else:
        make_ex_fn = make_cum_wt_example

    tmp = TFParquetDataset(
        train_paths, schema, shuffle_files=shuffle
    ).create_parallel_tf_dataset(
        batch_size, num_workers, num_epochs=args.repeat, queue_size=16, v2=True
    )
    train_dataset = tmp.shuffle(buffer_size=1000).map(make_ex_fn)

    validation_dataset_dict = {}
    # ts = int(time.time())
    ts = 1
    for country in train_countries:
        for eval_dt in get_dates_list_forwards(evaluate_date, args.eval_days):
            country_validation_dataset = (
                TFParquetDataset(
                    [
                        data_path(
                            TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_EXAMPLE_D7, country
                        )
                        % (train_variant, eval_dt)
                    ],
                    schema,
                    shuffle_files=False,
                )
                .create_tf_dataset(batch_size, num_epochs=1)
                .map(make_ex_fn)
                .take(validation_steps)
            )
            validation_dataset_dict[
                country + "-" + eval_dt
            ] = country_validation_dataset.cache(
                f"val_cumwatch_{country}_{args.all_wt}_{eval_dt}_{ts}"
            )

            """
            country_newcid_validation_dataset = (
                TFParquetDataset(
                    [
                        data_path(
                            TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_EXAMPLE_D7_NEWCID,
                            country,
                        )
                        % eval_dt
                    ],
                    schema,
                    shuffle_files=False,
                )
                .create_tf_dataset(batch_size, num_epochs=1)
                .map(make_ex_fn)
                .take(validation_steps)
            )
            validation_dataset_dict[
                country + "-new-" + eval_dt
            ] = country_newcid_validation_dataset.cache(
                f"val_cumwatch_new_{country}_{eval_dt}_{ts}"
            )
            """
    """
    evaluate_datasets = {
        f"{country}-{dt}": TFParquetDataset(
            [
                data_path(TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_EXAMPLE_D7, country)
                % (train_variant, dt)
            ],
            schema,
            shuffle_files=False,
        )
        .create_parallel_tf_dataset(batch_size, num_workers, v2=True)
        .map(make_ex_fn)
        for country in train_countries
        for dt in get_dates_list_forwards(evaluate_date, args.eval_days)
    }
    evaluate_datasets.update(
        {
            f"{country}-new-{dt}": TFParquetDataset(
                [
                    data_path(
                        TpfyDataPath.S3_CUMULATIVE_WATCH_TRAIN_EXAMPLE_D7_NEWCID,
                        country,
                    )
                    % dt
                ],
                schema,
                shuffle_files=False,
            )
            .create_parallel_tf_dataset(batch_size, num_workers, v2=True)
            .map(make_ex_fn)
            for country in train_countries
            for dt in get_dates_list_forwards(evaluate_date, args.eval_days)
        }
    )
    """

    session = tfv1.keras.backend.get_session()
    session.run(tfv1.tables_initializer())

    trainer = Trainer(model, session)

    if args.lr_decay:
        lr_scheduler = create_exp_lr_schedule_callback(
            20, 1e-4, 0.7 ** (1 / 10), verbose=args.verbose
        )
    else:
        lr_scheduler = None

    # if args.patience > 0:
    #     early_stopping = EarlyStopping(
    #         monitor="in_val_loss", patience=args.patience, mode="min", verbose=True
    #     )
    # else:
    #     early_stopping = None
    early_stopping = None

    print("start training")
    trained_epochs = trainer.train(
        train_dataset,
        epochs=train_max_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data_dict=validation_dataset_dict,
        validation_steps=validation_steps,
        validation_freq=10,
        log_dir="train/logs/" + model_name,
        lr_scheduler=lr_scheduler,
        early_stopping=None,
        verbose=args.verbose,
    )
    """
    eval_metrics = {}
    for eval_key, eval_ds in evaluate_datasets.items():
        eval_metrics[eval_key] = {
            k: float(v) for k, v in trainer.evaluate(eval_ds).items()
        }
    print(eval_metrics)
    os.makedirs("train/eval/", exist_ok=True)
    with open(f"train/eval/{model_name}.eval", "w") as f:
        json.dump(eval_metrics, f)
    """

    timestamp = int(time.time())
    export_path = "export/{}/{}".format(model_name, timestamp)
    print("export path", export_path)
    # predict func
    with tf.name_scope("serve"):
        predict_inputs = {}
        compact_predict_inputs = {}
        for feature in schema.features:
            predict_inputs[feature.name] = tfv1.placeholder(
                dtype=feature.tf_dtype,
                shape=[None, *feature.shape],
                name=feature.name,
            )
            if feature.name != "day":
                compact_batch_dim = 1 if feature.compact_predict else None
                compact_predict_inputs[feature.name] = tfv1.placeholder(
                    dtype=feature.tf_dtype,
                    shape=[compact_batch_dim, *feature.shape],
                    name=feature.name,
                )
        predict_outputs = model(predict_inputs, training=False)
        compact_predict_outputs = model(
            compact_predict_inputs, training=False, compact_predict=True
        )

    tfv1.add_to_collection(
        tfv1.saved_model.constants.MAIN_OP_KEY, tfv1.tables_initializer()
    )
    builder = SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        session,
        [SERVING],
        signature_def_map={
            "predictor": build_signature(predict_inputs, predict_outputs),
            "compact_predictor": build_signature(
                compact_predict_inputs, compact_predict_outputs
            ),
        },
        strip_default_attrs=True,
    )
    builder.save()

    print("write warmup requests")
    os.makedirs(f"{export_path}/assets.extra/", exist_ok=True)
    with tf.io.TFRecordWriter(
        f"{export_path}/assets.extra/tf_serving_warmup_requests"
    ) as writer:
        warmup_it = next(
            iter(validation_dataset_dict.values())
        ).make_one_shot_iterator()
        warmup_next = warmup_it.get_next()
        for i in range(1):
            features, _, _ = session.run(warmup_next)

            # pilot run
            feed_dict = {}
            dump_dict = {}
            for input_name, input_placeholder in compact_predict_inputs.items():
                if input_placeholder.shape[0] == 1:
                    value = features[input_name][:1]
                else:
                    value = features[input_name]
                feed_dict[input_placeholder] = value
                dump_dict[input_name] = tf.make_tensor_proto(value)
            session.run(compact_predict_outputs, feed_dict=feed_dict)

            request = predict_pb2.PredictRequest(
                inputs=dump_dict,
                model_spec=model_pb2.ModelSpec(signature_name="compact_predictor"),
            )
            log = prediction_log_pb2.PredictionLog(
                predict_log=prediction_log_pb2.PredictLog(request=request)
            )
            writer.write(log.SerializeToString())

    if args.upload:
        ts = int(time.time())
        export_s3_path_prefix = (
            model_path(TpfyDataPath.S3_CUMULATIVE_WATCH_MODEL_EXPORT, TENANT)
            % args.model_name
        )

        export_s3_path = "%s/export/%d" % (export_s3_path_prefix, ts)
        upload_folder(export_s3_path, export_path, set_acl=False)

        checkpoint_s3_path = "%s/checkpoint" % export_s3_path_prefix
        checkpoint_path = "%s/checkpoint" % export_path
        with open(checkpoint_path, "w") as checkpoint:
            checkpoint.write(str(ts))
        upload_file(checkpoint_s3_path, checkpoint_path, set_acl=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tray rank offline Training.")
    parser.add_argument("model_name", type=str)
    parser.add_argument("train_date", type=str)
    parser.add_argument("eval_date", type=str)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--train_days", type=int, default=0)
    parser.add_argument("--eval_days", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--all_wt", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--train_variant", type=str, default="")
    parser.add_argument(
        "--countries",
        type=str,
        help="countries to run, separated with comma. "
        "default is None. fallback to region countries",
    )
    args = parser.parse_args()

    run(args)
