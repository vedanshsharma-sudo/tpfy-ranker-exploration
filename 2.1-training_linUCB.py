import os
os.environ['ENV'] = 'prod'
os.environ['REGION'] = 'apse1'
os.environ['TENANT'] ="in"
os.environ['RECO_S3_BUCKET'] = "p13n-reco-offline-prod"
os.environ['COUNTRY_KEY']= "in"
os.environ['AWS_REGION']= "ap-southeast-1"
os.environ['USE_REAL_CMS3']= "True"
os.environ['RECO_CREDENTIAL']= "-----BEGINRSAPRIVATEKEY-----\nMGICAQACEQCdHOlGnxIMWCMzjK2JAg37AgMBAAECEGOIwGTEO9vd3X9+jyiF4NECCQnoqDakDgSm2QIID9sadWN0XvMCCQLiqPkgVKSuIQIIDCAsWM+pJB8CCQG0jbIGCNX9MA==\n-----ENDRSAPRIVATEKEY-----"

import argparse, gc
import json
import os, time
import numpy as np
import s3fs
import pyarrow
import tensorflow as tf
from tqdm import tqdm

tfv1 = tf.compat.v1
tfv1.disable_v2_behavior()

# Enable memory growth for GPUs to avoid memory fragmentation
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

import tensorflow_addons as tfa
import tensorflow_recommenders_addons as tfra
from model.losses import masked_binary_entropy_loss
from model.metrics import MaskedAUC

from common.config.utils import data_path, model_path
from common.config import TENANT
from tpfy.tf_model.tpfy_model_v3_mtl import TpfyModelV3, TpfyMtlModelConfig
from tpfy.etl.schema import TpfyMtlDatasetSchema
from model.parquet_dataset import TFParquetDataset
from tpfy.common import TpfyDataPath
from omegaconf import OmegaConf
from dataclasses import dataclass
from tpfy.train_v3_mtl import make_example_mtl, TpfyTrainConfig, TpfyConfig
from tpfy.helper import load_model_weights_from_s3

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

session = tfv1.keras.backend.get_session()

def create_training_dataset(date):
    data_path_str = data_path(
        TpfyDataPath.S3_TPFY_IMPR_V3_AGG_MTL_EXTRACTED_EXAMPLES_VAR, TENANT
    ) % (variant, date)

    dataset = TFParquetDataset([data_path_str], TpfyMtlDatasetSchema, shuffle_files=True)
    tf_dataset = dataset.create_tf_dataset(batch_size).map(make_example_mtl)
    # tf_dataset = dataset.create_parallel_tf_dataset(
    #     batch_size,
    #     args.num_workers,
    #     num_epochs=1,
    #     queue_size=16,
    #     v2=True,
    #     row_transformer_factory=None,
    # ).map(make_example_mtl)
    # train_it = tfv1.data.make_initializable_iterator(tf_dataset)
    return tf_dataset

def load_and_compile_model():
    # Build model
    print(f"\n{'='*80}")
    print("BUILDING MODEL")
    print(f"{'='*80}")

    model = TpfyModelV3(
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

    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

    model.compile(optimizer=optimizer, loss=loss_dict, metrics=metric_dict)
    print("Model compiled")
    
    return model

def get_activations_and_labels(iterator, model, last_layer_tensor):
    features, labels, metadata = session.run(iterator)

    # Run model
    predictions = model(features, training=False)

    # Execute
    pred_values, activation_values = session.run(
        [predictions, last_layer_tensor]
    )

    return activation_values, labels, pred_values, metadata

def compute_A(A, iterator, model, last_layer_tensor):
    H, y_batch_all_labels, _, _ = get_activations_and_labels(iterator, model, last_layer_tensor)
    y_batch = y_batch_all_labels['click']

    mask = (y_batch != -1)

    if not np.any(mask):
        return A

    H = H[mask.squeeze()]
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    # Accumulate
    A += H.T @ H

    assert np.linalg.eigvalsh(A).min() > 0
    return A

if __name__ == '__main__':
    tf_dataset = create_training_dataset(args.date)
    iterator = tf_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    sample_features, sample_labels, sample_metadata = session.run(next_batch)
    tpfy_model = load_and_compile_model()

    prediction = tpfy_model(sample_features, training=False)
    session.run([
        tfv1.global_variables_initializer(),
        tfv1.local_variables_initializer(),
        tfv1.tables_initializer()
    ])

    plain_weights = load_model_weights_from_s3(
        args.model_name,
        use_s3=True
    )
    plain_weights_modified = {k.replace('train/', ''): v for k, v in plain_weights.items()}
    restore_ops = tpfy_model.restore_plain_weights_ops(
        plain_weights_modified,
        clear_nn=args.clear_nn
    )
    session.run(restore_ops)

    # Create NEW iterator (reset to start of dataset)
    iterator = tf_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    # Get compress_output tensor (linear_input)
    graph = tf.compat.v1.get_default_graph()
    compress_output_tensor = graph.get_tensor_by_name('tpfy_model_v3/deepfm/Relu:0')

    #train feature matrix
    lambda_=1.0
    d=128
    A = lambda_ * np.eye(d, dtype=np.float64)

    start = time.time()
    for run_ in range(10_000):
        if (run_ % 100 == 0) and (run):
            np.save(f'tpfy/neural_linUCB_training_data/A_{run_}.npy', A)
            print(f'Run {run_} completed in {time.time() - start} s!')
            start = time.time()
        A = compute_A(A, next_batch, tpfy_model, compress_output_tensor)