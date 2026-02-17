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

## python3.7 -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-09 --checkpoint 1770723470
import argparse, gc
import os, time, shutil
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
from tpfy.common import TpfyDataPath
from tpfy.train_v3_mtl import TpfyConfig
from tpfy.helper import load_model_weights_from_s3, create_dataset, save_matrices_to_s3, load_matrices_from_s3_direct

S3_TPFY_NEURAL_LINUCB_MODEL_EXPORT = model_path(TpfyDataPath.S3_TPFY_NEURAL_LINUCB_MATRICES, TENANT)

def load_and_compile_model(args, hparams):
    """
    Docstring for load_and_compile_model
    
    :param args: hyperparameters and arguments for model loading and compilation
    :param hparams: hyperparameters and arguments for model loading and compilation
    """
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
    """
    Docstring for compute_A_b
    
    :param A: Covariance matrix A to be updated : (B, d, d)
    :param b: Response vector b to be updated : (B, d)
    :param next_batch: iterator batch containing features, labels, and metadata tensors
    :param last_layer_tensor: Activation tensor from the last layer of the neural network to be used as context features for LinUCB
    :param session: TensorFlow session for running the computation
    """
    features, labels, metadata = next_batch
    
    features, labels, _, activation_values = session.run(
        [features, labels, metadata, last_layer_tensor]
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

    # A is computed using the outer product of the features (H) and b is computed using the product of features 
    # and rewards (y_batch)
    A += H.T @ H
    b += H.T @ y_batch

    del H, labels, y_batch, features, metadata
    return A, b

def run(args, session, hparams):
    variant = args.variant
    if variant and not variant.startswith("-"):
        variant = "-" + variant
    
    batch_size = args.batch_size or hparams.train.batch_size
    
    tf_dataset = create_dataset(args.date, variant=variant, batch_size=batch_size)
    tpfy_model = load_and_compile_model(args=args, hparams=hparams)

    with tfv1.name_scope("linucb_training"):
        iterator = tf_dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()

        # Unpack - this creates tensor placeholders
        if len(next_batch) == 3:
            x, y_true, metadata = next_batch
        else:
            x, y_true = next_batch
            metadata = None

        y_pred = tpfy_model(x, training=False)

    session.run([
        tfv1.global_variables_initializer(),
        tfv1.local_variables_initializer(),
        tfv1.tables_initializer()
    ])

    # last trained DeepFM model checkpoint is loaded here to extract last layer representations as features for LinUCB.
    plain_weights = load_model_weights_from_s3(
        args.model_name,
        use_s3=True,
        checkpoint_name=args.checkpoint
    )

    # The keys in plain_weights have the format 'train/layer_name/weight_name'. 
    # We need to modify them to match the current model's scope, which is 'linucb_training/tpfy_model_v3/deepfm/layer_name/weight_name'
    plain_weights_modified = {k.replace('train/', 'linucb_training/'): v for k, v in plain_weights.items()}
    restore_ops = tpfy_model.restore_plain_weights_ops(
        plain_weights_modified,
        clear_nn=args.clear_nn
    )
    session.run(restore_ops)

    # Create NEW iterator (reset to start of dataset)
    iterator = tf_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    # Getting the tensor for the specified layer's activations to be used as features for LinUCB
    # Here, extracting the output Relu(SparseFeaturesOutput + DenseFeaturesOutput). This is given to Linear layer(d, 2) to predict click and watch logits.
    graph = tfv1.get_default_graph()
    compress_output_tensor = graph.get_tensor_by_name(
        f'linucb_training/tpfy_model_v3/deepfm/{args.layer_name}:0'
    )
    d = hparams.model.middle_dim
    lambda_ = args.lambda_reg
        
    # s3_base_path = S3_TPFY_NEURAL_LINUCB_MODEL_EXPORT
    s3_base_path = 's3://p13n-reco-offline-prod/upload_objects/test_vedansh'
    base_path = f'export/neural_linUCB_offline_matrices_{args.date}'
    os.makedirs(base_path, exist_ok=True)
    
    # Load or initialize matrices
    if args.reset_matrix:
        print("Resetting matrices to default initialization")
        A = lambda_ * np.eye(d, dtype=np.float32)
        b = np.zeros((d,), dtype=np.float32)
    else:
        print("Loading matrices from previous run")
        s3_latest_path = f"{s3_base_path}/latest_matrices"
        A, b = load_matrices_from_s3_direct(s3_latest_path)

    start = time.time()
    run_ = 0
    while True:
        if (run_ % args.logging_steps == 0) and (run_):
            np.save(f'{base_path}/A.npy', A)
            np.save(f'{base_path}/b.npy', b)
            np.save(f'{base_path}/A_inv.npy', np.linalg.inv(A))
            # Upload entire folder to S3 if enabled
            if args.upload:
                save_matrices_to_s3(f"{s3_base_path}/{args.date}", A, b)
                print(f"Uploaded folder to S3: {s3_base_path}/{args.date}/")

            gc.collect()
            print(f'Run {run_} completed in {time.time() - start} s!')
            start = time.time()
            
        try:
            A, b = compute_A_b(A, b, next_batch, compress_output_tensor, session=session)
        except tf.errors.OutOfRangeError:
            print(f"\nDataset ended at run {run_}. Ending training loop.")
            break
        
        run_ += 1

    # Final save
    np.save(f'{base_path}/A.npy', A)
    np.save(f'{base_path}/b.npy', b)
    np.save(f'{base_path}/A_inv.npy', np.linalg.inv(A))
    if args.upload:
        save_matrices_to_s3(f"{s3_base_path}/{args.date}", A, b)
        save_matrices_to_s3(f"{s3_base_path}/latest_matrices", A, b)
        print(f"Final upload to S3: {s3_base_path}")

    print("\nTraining completed successfully!")
    print(f"Final A matrix shape: {A.shape}")
    print(f"Final b vector shape: {b.shape}")

def main():
    parser = argparse.ArgumentParser(description="TPFY Exploration offline Training.")
    parser.add_argument("model_name", type=str, help="TPFY DeepFM model name in S3 for weight initialization")
    parser.add_argument("date", type=str, help="Training date in YYYY-MM-DD format (used for dataset path and S3 saving)")
    parser.add_argument("--click_ns", type=float, default=0.08, help="Negative sampling ratio for clicks")
    parser.add_argument("--lambda_reg", type=float, default=1, help="Regularization parameter for covariance matrix A")
    parser.add_argument("--variant", type=str, default="cms3", help="CMS Dataset variant to use")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--clear_nn", action="store_true", default=False, help="Clear neural network weights before training")
    parser.add_argument("--checkpoint", default=None, type=str, help="Specific DeepFM checkpoint to load for model initialization, otherwise latest checkpoint is used")
    parser.add_argument("--layer_name", default='Relu', type=str, help="Activations from which layer to use for LinUCB (default: 'Relu' - the last activation layer before output)")
    parser.add_argument("--upload", action="store_true", help="Whether to upload the matrices to S3")
    parser.add_argument("--logging_steps", default=1000, type=int, help="Number of steps between logging and saving matrices to S3")
    parser.add_argument("--reset_matrix", action="store_true", default=False, help="Whether to reset matrices to default initialization")

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

    session = tfv1.keras.backend.get_session()
    print("Start training")
    run(args, session, hparams)

if __name__ == "__main__":
    main()