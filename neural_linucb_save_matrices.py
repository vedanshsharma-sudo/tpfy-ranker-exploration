# python3.7 -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-09 --checkpoint 1770723470
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
import os, time
import numpy as np
import tensorflow as tf

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
from tpfy.tf_model.tpfy_model_v3_mtl import TpfyModelV3
from omegaconf import OmegaConf
from tpfy.train_v3_mtl import TpfyConfig
from tpfy.helper import load_model_weights_from_s3, create_dataset, save_matrices
from common.s3_utils import upload_folder

def load_and_compile_model(args, hparams):
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

def compute_A_b(A, b, next_batch, last_layer_tensor, session):
    features, labels, _ = session.run(next_batch)
    
    activation_values = session.run(
        last_layer_tensor,
        feed_dict={} if not features else None  
    )
    y_batch = labels['click']
    mask = (y_batch != -1)

    if not np.any(mask):
        del activation_values, labels, mask
        return A, b

    H = activation_values[mask.squeeze()].copy() 
    y_batch = y_batch[mask].reshape(sum(mask)[0], )
    
    del activation_values, mask  # Delete immediately
    
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    A += H.T @ H
    b += H.T @ y_batch

    del H, labels, y_batch
    return A, b

def run(args, session, hparams):
    variant = args.variant
    if variant and not variant.startswith("-"):
        variant = "-" + variant
    
    batch_size = args.batch_size or hparams.train.batch_size
    
    tf_dataset = create_dataset(args.date, variant=variant, batch_size=batch_size)
    iterator = tf_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    sample_features, _, _ = session.run(next_batch)
    tpfy_model = load_and_compile_model(args=args, hparams=hparams)

    _ = tpfy_model(sample_features, training=False)
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
    
    d = hparams.model.middle_dim
    lambda_ = args.lambda_reg
    A = lambda_ * np.eye(d, dtype=np.float32)
    b = np.zeros((d,), dtype=np.float32)
    base_path = f'export/neural_linUCB_offline_matrices_{args.date}'
    
    # S3 path (only if upload is enabled)
    # s3_base_path = model_path(
    # f"tpfy/tpfy-v3-neural-linucb/{args.date}",
    # TENANT
    # )
    
    s3_base_path = 's3://p13n-reco-offline-prod/upload_objects/test_vedansh/'

    start = time.time()
    run_ = 0
    os.makedirs(base_path, exist_ok=True)
    while True:
        if (run_ % args.logging_steps == 0) and (run_):
            save_matrices(base_path, A, b)

            # Upload entire folder to S3 if enabled
            if args.upload:
                upload_folder(s3_base_path, base_path, set_acl=False)
                print(f"Uploaded folder to S3: {s3_base_path}")

            gc.collect()
            print(f'Run {run_} completed in {time.time() - start} s!')
            start = time.time()
            
        try:
            A, b = compute_A_b(A, b, next_batch, compress_output_tensor, session=session)
        except Exception as e:
            print(f"Ended due to exception : {e}")
            # Final save to local
            save_matrices(base_path, A, b)
            
            # Final upload entire folder to S3
            if args.upload:
                upload_folder(s3_base_path, base_path, set_acl=False)
                print(f"Final upload to S3: {s3_base_path}")
            break
        run_ += 1
        
def main():
    parser = argparse.ArgumentParser(description="TPFY Exploration offline Training.")
    parser.add_argument("model_name", type=str)
    parser.add_argument("date", type=str)
    parser.add_argument("--click_ns", type=float, default=0.08)
    parser.add_argument("--lambda_reg", type=float, default=1)
    parser.add_argument("--variant", type=str, default="cms3")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--clear_nn", action="store_true", default=False)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--layer_name", default='Relu', type=str)
    parser.add_argument("--upload", action="store_true", help="uploading model to s3")
    parser.add_argument("--logging_steps", default=1000, type=int)

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
    run(args, session, hparams)

if __name__ == "__main__":
    main()