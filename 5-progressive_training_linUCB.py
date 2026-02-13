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

def create_dataset(date, path = None):
    if path:
        data_path_str = path
    else:
        data_path_str = data_path(
            TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_MTL_EXTRACTED_EXAMPLES, TENANT
        ) % (variant, date)

    dataset = TFParquetDataset([data_path_str], TpfyMtlDatasetSchema, shuffle_files=True)
    tf_dataset = dataset.create_tf_dataset(batch_size).map(make_example_mtl)
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

def get_activations_and_labels(next_batch, last_layer_tensor):
    features, labels, metadata = session.run(next_batch)
    
    activation_values = session.run(
        last_layer_tensor,
        feed_dict={} if not features else None  
    )
    
    return activation_values, labels, metadata

def compute_A(A, next_batch, last_layer_tensor):
    activation_values, labels, metadata = get_activations_and_labels(next_batch, last_layer_tensor)
    y_batch = labels['click']
    mask = (y_batch != -1)

    if not np.any(mask):
        del activation_values, labels, metadata, mask
        return A

    H = activation_values[mask.squeeze()].copy() 
    del activation_values, mask  # Delete immediately
    
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    A += H.T @ H

    del H, labels, metadata 
    return A

def run(args):
    tf_dataset = create_dataset(args.date)
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
        use_s3=True,
        checkpoint_name=args.checkpoint
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
    compress_output_tensor = graph.get_tensor_by_name(f'tpfy_model_v3/deepfm/{args.layer_name}:0')

    #train feature matrix
    # lambda_=1.0
    d=128
    # A = lambda_ * np.eye(d, dtype=np.float64)
    A = np.zeros((d,d), dtype=np.float64)

    start = time.time()
    run_ = 0
    os.makedirs(f'tpfy/neural_linUCB_offline_matrices_{args.date}_{args.checkpoint}', exist_ok=True)
    while True:
        if (run_ % 100 == 0) and (run_):
            np.save(f'tpfy/neural_linUCB_offline_matrices_{args.date}_{args.checkpoint}/A_{run_}.npy', A)
            gc.collect()
            print(f'Run {run_} completed in {time.time() - start} s!')
            start = time.time()
        try:
            A = compute_A(A, next_batch, compress_output_tensor)
        except tf.errors.OutOfRangeError:
            print("End of dataset reached")
            break
        run_ += 1
    
    
parser = argparse.ArgumentParser(description="TPFY Exploration offline Training.")
parser.add_argument("model_name", type=str)
parser.add_argument("date", type=str)
parser.add_argument("--click_ns", type=float, default=0.08)
parser.add_argument("--variant", type=str, default="cms3")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--clear_nn", action="store_true", default=False)
parser.add_argument("--checkpoint", default=None, type=str)
parser.add_argument("--layer_name", default='Relu', type=str)

args = parser.parse_args()

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
print("Start training")
run(args)