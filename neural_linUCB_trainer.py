import argparse, gc
import json
import os
import numpy as np
import s3fs
import pyarrow
import tensorflow as tf
from tqdm import tqdm

tfv1 = tf.compat.v1
tfv1.disable_v2_behavior()

import tensorflow_addons as tfa
import tensorflow_recommenders_addons as tfra

from common.config.utils import data_path, model_path
from common.config import TENANT
from tpfy.tf_model.tpfy_model_v3_mtl import TpfyModelV3, TpfyMtlModelConfig
from tpfy.etl.schema import TpfyMtlDatasetSchema
from model.parquet_dataset import TFParquetDataset
from tpfy.common import TpfyDataPath
from omegaconf import OmegaConf
from dataclasses import dataclass
from tpfy.train_v3_mtl import make_example_mtl, TpfyTrainConfig, TpfyConfig

S3_TPFY_MODEL_EXPORT = model_path(TpfyDataPath.S3_TPFY_MODEL_EXPORT, TENANT)

def load_model_weights_from_s3(model_name, use_s3=True):
    """
    Load plain weights from S3 or local filesystem
    
    Args:
        model_name: Name of the model (e.g., "my_model/12345678")
        checkpoint: Specific checkpoint to load (None = read from checkpoint file)
        use_s3: If True, load from S3; if False, load from local export/
    
    Returns:
        dict: Plain weights dictionary
    """
    if use_s3:
        filesystem = s3fs.S3FileSystem(use_ssl=False)
        model_path = S3_TPFY_MODEL_EXPORT % model_name
    else:
        filesystem = pyarrow.LocalFileSystem()
        model_path = os.path.join("export", model_name)
    
    # Read checkpoint if not specified
    checkpoint_path = os.path.join(model_path, "checkpoint")
    print(f"Reading checkpoint from: {checkpoint_path}")

    if not filesystem.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    with filesystem.open(checkpoint_path, "r") as f:
        checkpoint = f.read().strip()
    print(f"Using checkpoint: {checkpoint}")
    
    # Load weights
    weights_path = os.path.join(model_path, checkpoint, "plain_weights.npz")
    print(f"Loading weights from: {weights_path}")
    
    if not filesystem.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    with filesystem.open(weights_path, "rb") as f:
        plain_weights = {}
        npz_data = np.load(f)
        for k, v in npz_data.items():
            plain_weights[k] = v
    
    print(f"Loaded {len(plain_weights)} weight tensors")
    print(f"Weight keys: {list(plain_weights.keys())[:10]}...")  # Show first 10
    
    return plain_weights

class Args:
    """Simple class to hold training arguments (replaces argparse)"""
    def __init__(self):
        # Positional arguments
        self.model_name = "tpfy-v3-mtl-r2"
        self.date = "2026-02-06"  # Training date
        self.val_date = "2026-02-06"  # Validation date
        
        # Optional arguments
        self.conf = None
        self.max_epoch = None
        self.val_days = 1
        self.click_ns = 0.08
        self.variant = "cms3"
        self.num_workers = 4
        self.repeat = 1
        self.eval_freq = None
        self.lr = 1e-4
        self.batch_size = 512
        self.click_weight = 1.0
        self.watch_weight = 1.0
        self.upload = False  # Set to False if you don't want to upload to S3
        self.reload_local_model = None
        self.reload_s3_model = "tpfy-v3-mtl-r2"  # Set to None if starting fresh
        self.extract_activations = True
        self.output = None
        self.clear_nn = False
        self.ckpt = None
        self.verbose = False
        self.countries = None

import pandas as pd
content_popularity_tag = pd.read_csv('/home/ubuntu/vedansh/code/rank-change-analysis/content_id_popularity_tag.csv')
content_popularity_tag.drop_duplicates(subset=['sub_title_id'], inplace = True)
content_popularity_tag.reset_index(drop = True, inplace = True)
content_popularity_dict = content_popularity_tag.set_index('sub_title_id')['popularity_tag'].to_dict()

from sklearn.metrics import mean_squared_error
import pickle
    
def run():
    print("="*80)
    print("TPFY MODEL INFERENCE FROM PLAIN WEIGHTS")
    print("="*80)

    # Load configuration
    config_name = f"tpfy/tpfy_config/mtl-{TENANT}.yaml"
    if not os.path.exists(config_name):
        raise FileNotFoundError(f"Config file {config_name} not found")

    hparams: TpfyConfig = OmegaConf.merge(
        OmegaConf.structured(TpfyConfig),
        OmegaConf.load(config_name),
    )
    print(f"\nLoaded config: {config_name}")

    # Override batch size if specified
    if args.batch_size:
        hparams.train.batch_size = args.batch_size

    batch_size = hparams.train.batch_size
    print(f"Batch size: {batch_size}")

    # Load dataset
    variant = args.variant
    if variant and not variant.startswith("-"):
        variant = "-" + variant

    data_date = args.date
    data_path_str = data_path(
        TpfyDataPath.S3_TPFY_IMPR_V3_AGG_MTL_EXTRACTED_EXAMPLES_VAR, TENANT
    ) % (variant, data_date)

    print(f"\nData path: {data_path_str}")

    dataset = TFParquetDataset([data_path_str], TpfyMtlDatasetSchema, shuffle_files=False)
    tf_dataset = dataset.create_tf_dataset(batch_size).map(make_example_mtl)

    # if args.num_batches:
    #     tf_dataset = tf_dataset.take(args.num_batches)
    #     print(f"Limiting to {args.num_batches} batches")

    # Create TensorFlow session
    session = tfv1.keras.backend.get_session()

    # Build model
    print(f"\n{'='*80}")
    print("BUILDING MODEL")
    print(f"{'='*80}")

    tpfy_model = TpfyModelV3(
        hparams.model,
        click_ns=args.click_ns,
        enable_random_watch=hparams.train.enable_random_watch,
    )

    # Create optimizer (needed for compilation, even though we won't train)
    optimizer = tfa.optimizers.AdamW(
        weight_decay=0.0,  # Not needed for inference
        learning_rate=0.001,  # Not needed for inference
        epsilon=1e-4,
    )
    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

    # Compile model (required to initialize variables)
    from model.losses import masked_binary_entropy_loss
    from model.metrics import MaskedAUC

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

    tpfy_model.compile(optimizer=optimizer, loss=loss_dict, metrics=metric_dict)
    print("Model compiled")

    # ========== BUILD MODEL WITH SAMPLE DATA ==========
    print("\nFetching sample batch to build model...")
    iterator = tf_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    sample_features, sample_labels, sample_metadata = session.run(next_batch)
    print(f"Sample batch loaded: {len(sample_features)} features")

    print("Running model forward pass to create weights...")
    _ = tpfy_model(sample_features, training=False)

    print(f"Model built successfully")
    print(f"Total trainable variables: {len(tpfy_model.trainable_variables)}")

    # Initialize all variables
    print("\nInitializing TensorFlow variables...")
    session.run([
        tfv1.global_variables_initializer(),
        tfv1.local_variables_initializer(),
        tfv1.tables_initializer(),
    ])
    print("✅ Variables initialized")
    # ==================================================

    # Load plain weights from S3
    print(f"\n{'='*80}")
    print("LOADING MODEL WEIGHTS")
    print(f"{'='*80}")

    plain_weights = load_model_weights_from_s3(
        args.model_name,
        use_s3=True
    )
    plain_weights_modified = {k.replace('train/', ''): v for k, v in plain_weights.items()}

    # Restore weights (NOW the model is built, so this will work)
    print("\nRestoring model weights...")
    restore_ops = tpfy_model.restore_plain_weights_ops(
        plain_weights_modified,
        clear_nn=args.clear_nn
    )
    session.run(restore_ops)
    print("Weights restored successfully")

    # Create NEW iterator (reset to start of dataset)
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE")
    print(f"{'='*80}")

    iterator = tf_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    # Get compress_output tensor (linear_input)
    graph = tf.compat.v1.get_default_graph()

    # Find the compress_out tensor that is fed to output layer that predicts click / watch
    compress_output_tensor = graph.get_tensor_by_name('tpfy_model_v3/deepfm/Relu:0')
    
    from common.time_utils import get_dates_list_forwards
    from model.trainer import Trainer, ValData, LearningRateScheduler

    validation_dataset_dict = {}
    dataset_schema = TpfyMtlDatasetSchema
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
                .map(make_example_mtl),
                active_objectives=[
                    "click",
                    "watch",
                    "random_watch",
                    "paywall_view",
                    "add_watchlist",
                ],
            )

    val_it = tfv1.data.make_initializable_iterator(validation_dataset_dict['in-2026-02-06'].tf_dataset)
    session.run(val_it.initializer)
    val_next = val_it.get_next()
    
    def get_activations_and_labels(iterator, model, last_layer_tensor):
        features, labels, metadata = session.run(iterator)

        # Run model
        predictions = model(features, training=False)

        # Execute
        pred_values, activation_values = session.run(
            [predictions, last_layer_tensor]
        )

        return activation_values, labels, pred_values, metadata

    def compute_A_b_tf(A, b, iterator, model, last_layer_tensor, logging = False):
        H, y_batch_all_labels, predictions, metadata = get_activations_and_labels(iterator, model, last_layer_tensor)
        y_batch = y_batch_all_labels['click']

        mask = (y_batch != -1)

        if not np.any(mask):
            return A, b

        H = H[mask.squeeze()]
        y_batch = y_batch[mask].reshape(sum(mask)[0], 1)

        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

        # Accumulate
        A += H.T @ H
        b += H.T @ y_batch

        assert A.shape == (d, d)
        assert b.shape == (d,1)
        assert np.allclose(A, A.T, atol=1e-6)
        assert np.linalg.eigvalsh(A).min() > 0

        if logging:
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            mean = H@theta
            mse_value = mean_squared_error(y_batch, mean)
            return A, b, mse_value

        return A, b, 0

    def validation(iterator, model, last_layer_tensor, d=128, runs=10):
        predictions = []
        labels = []
        variances = []
        deepFM_predictions = []
        content_ids = []
        masks = []
        mse_values = []

        for i in tqdm(range(runs)):
            H, y_batch_all_labels, pred_values, metadata = get_activations_and_labels(iterator, model, last_layer_tensor)
            y_batch = y_batch_all_labels['click']
            pred_values = pred_values['click']

            mask = (y_batch != -1)

            if not np.any(mask):
                continue

            H = H[mask.squeeze()]
            # pred_values = pred_values[mask.squeeze()]
            y_batch = y_batch[mask].reshape(sum(mask)[0], 1)
            pred_values = pred_values[mask].reshape(sum(mask)[0], 1)

            H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
            variance = np.sqrt(H @ A_inv @ H.T)
            mean = H@theta

            predictions.append(mean)
            labels.append(y_batch)
            variances.append(variance)
            deepFM_predictions.append(pred_values)
            content_ids.append([content_id for content_id, mask_bool in zip(metadata['content_id'], mask) if mask_bool])
            mse_values.append(mean_squared_error(y_batch, mean))

        return predictions, labels, variances, deepFM_predictions, content_ids, sum(mse_values)/len(mse_values)

    lambda_=1.0
    d=128
    A = lambda_ * np.eye(d, dtype=np.float64)
    b = np.zeros((d, 1), dtype=np.float64)

    run = 0
    
    # training starts here
    while True:
        if (run % 100 == 0) and (run):
            #find the training loss for the current batch
            A, b, mse_value = compute_A_b_tf(A, b, next_batch, tpfy_model, compress_output_tensor, logging=True)

            #find the validation evals for the 1000 runs
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            predictions, labels, variances, deepFM_predictions, content_ids, val_mse = validation(val_next, tpfy_model, compress_output_tensor, runs = 10)
            all_labels = np.concatenate(labels, axis=None)
            all_predictions = np.concatenate(predictions, axis=None)
            # all_deepFMpredictions = np.concatenate(deepFM_predictions, axis=None)
            all_variances = np.concatenate([np.diag(x) for x in variances], axis = None)
            all_contentIds = [x for content_id in content_ids for x in content_id]

            content_variances = {int(content_id): variances for content_id, variances in zip(all_contentIds, all_variances)}
            content_variances_df = pd.DataFrame(content_variances, index = [0]).T.reset_index().rename(columns = {'index': 'content_id', 0: 'variances'})
            content_variances_df['popularity_tag'] = content_variances_df['content_id'].apply(lambda x: content_popularity_dict.get(int(x)))
            popularity_variance_dist_stats = content_variances_df.groupby('popularity_tag')['variances'].describe().to_dict()

            #content popularity tier vs variancess
            content_predictions = {int(content_id): predictions for content_id, predictions in zip(all_contentIds, all_predictions)}
            content_predictions_df = pd.DataFrame(content_predictions, index = [0]).T.reset_index().rename(columns = {'index': 'content_id', 0: 'predictions'})
            content_predictions_df['popularity_tag'] = content_predictions_df['content_id'].apply(lambda x: content_popularity_dict.get(int(x)))
            popularity_mean_dist_stats = content_predictions_df.groupby('popularity_tag')['predictions'].describe().to_dict()

            dumping_dict = {}
            dumping_dict['train_mse'] = mse_value
            dumping_dict['valid_mse'] = val_mse
            dumping_dict['valid_popularity_variance_dist_stats'] = popularity_variance_dist_stats
            dumping_dict['valid_popularity_mean_dist_stats'] = popularity_mean_dist_stats

            with open(f'tpfy/neural_linUCB_training_data/training_stats_run_{run}.pkl', 'wb') as handle:
                pickle.dump(dumping_dict, handle)

            np.save(f'tpfy/neural_linUCB_training_data/A_inv_{run}.npy', A_inv)
            np.save(f'tpfy/neural_linUCB_training_data/b_{run}.npy', b)

            print(f'Run {run} completed!')
            gc.collect()
        else:
            A, b, _ = compute_A_b_tf(A, b, next_batch, tpfy_model, compress_output_tensor)
        run += 1
        print(run)
        
if __name__ == '__main__':
    # Create args instance
    args = Args()
    # Display configuration
    print("Training Configuration:")
    print(f"  Model Name: {args.model_name}")
    print(f"  Training Date: {args.date}")
    print(f"  Validation Date: {args.val_date}")
    print(f"  Variant: {args.variant}")
    print(f"  Click NS: {args.click_ns}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  Reload Model: {args.reload_s3_model}")
    print(f"  Upload: {args.upload}")
    run()