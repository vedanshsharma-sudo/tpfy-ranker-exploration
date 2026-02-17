import s3fs, os
import pyarrow
import numpy as np
from common.config.utils import data_path, model_path
from tpfy.common import TpfyDataPath
from common.config import TENANT
from model.parquet_dataset import TFParquetDataset
from tpfy.common import TpfyDataPath
from tpfy.etl.schema import TpfyMtlDatasetSchema
from tpfy.train_v3_mtl import make_example_mtl
import os, io, boto3
from common.s3_utils import download_all_files

S3_TPFY_MODEL_EXPORT = model_path(TpfyDataPath.S3_TPFY_MODEL_EXPORT, TENANT)

def create_dataset(date, variant, batch_size, path = None):
    if path:
        data_path_str = path
    else:
        data_path_str = data_path(
            TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_MTL_EXTRACTED_EXAMPLES, TENANT
        ) % (variant, date)

    dataset = TFParquetDataset([data_path_str], TpfyMtlDatasetSchema, shuffle_files=True)
    tf_dataset = dataset.create_tf_dataset(batch_size).map(make_example_mtl)
    return tf_dataset

def save_matrices_to_s3(s3_path, A, b):
    """Save matrices directly to S3 without local storage."""
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    bucket_name = s3_path.split('/')[0]
    prefix = '/'.join(s3_path.split('/')[1:])
    
    # Save A matrix
    A_buffer = io.BytesIO()
    np.save(A_buffer, A)
    A_buffer.seek(0)
    s3_client.upload_fileobj(A_buffer, bucket_name, f"{prefix}/A.npy")
    print(f"Uploaded A matrix to s3://{bucket_name}/{prefix}/A.npy")
    
    # Save b vector
    b_buffer = io.BytesIO()
    np.save(b_buffer, b)
    b_buffer.seek(0)
    s3_client.upload_fileobj(b_buffer, bucket_name, f"{prefix}/b.npy")
    print(f"Uploaded b vector to s3://{bucket_name}/{prefix}/b.npy")
    
    A_inv = np.linalg.inv(A)
    A_inv_buffer = io.BytesIO()
    np.save(A_inv_buffer, A_inv)
    A_inv_buffer.seek(0)
    s3_client.upload_fileobj(A_inv_buffer, bucket_name, f"{prefix}/A_inv.npy")
    print(f"Uploaded A inverse matrix to s3://{bucket_name}/{prefix}/A_inv.npy")
    
    A_buffer.close()
    A_inv_buffer.close()
    b_buffer.close()

def load_matrices_from_s3_direct(s3_path):
    """Load matrices directly from S3 without local download."""
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    bucket_name = s3_path.split('/')[0]
    prefix = '/'.join(s3_path.split('/')[1:])
    
    # try:
        # Load A matrix
    A_buffer = io.BytesIO()
    s3_client.download_fileobj(bucket_name, f"{prefix}/A.npy", A_buffer)
    A_buffer.seek(0)
    A = np.load(A_buffer)
    A_buffer.close()

    # Load b vector
    b_buffer = io.BytesIO()
    s3_client.download_fileobj(bucket_name, f"{prefix}/b.npy", b_buffer)
    b_buffer.seek(0)
    b = np.load(b_buffer)
    b_buffer.close()

    print(f"Loaded matrices from s3://{bucket_name}/{prefix}/")
    return A, b

def load_model_weights_from_s3(model_name, use_s3=True, checkpoint_name = None):
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
    
    if checkpoint_name is None:
        with filesystem.open(checkpoint_path, "r") as f:
            checkpoint = f.read().strip()
        print(f"Using checkpoint: {checkpoint}")
    else:
        checkpoint = checkpoint_name
    
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

def save_matrices(base_path, A, b):
    """Save A, b, and A_inv matrices to disk."""
    os.makedirs(base_path, exist_ok=True)
    np.save(f'{base_path}/A.npy', A)
    np.save(f'{base_path}/b.npy', b)
    
    np.save(f'{base_path}/A_inv.npy', np.linalg.inv(A))
    
    print(f"Saved matrices to {base_path}")

def load_matrices_from_s3(s3_path, local_path, d):
    """
    Load A and b matrices from S3.
    Raises error if matrices cannot be loaded (when not in reset mode).
    """
    # Download matrices from S3
    os.makedirs(local_path, exist_ok=True)
    
    try:
        download_all_files(s3_path, local_path, clean=True, reserve_structure=False)
    except Exception as e:
        raise RuntimeError(f"Failed to download matrices from S3 path {s3_path}: {e}")
    
    A_path = f'{local_path}/A.npy'
    b_path = f'{local_path}/b.npy'
    
    # Assert files exist
    assert os.path.exists(A_path), f"A.npy not found in {local_path}. Use --reset_matrix for first run."
    assert os.path.exists(b_path), f"b.npy not found in {local_path}. Use --reset_matrix for first run."
    
    try:
        A = np.load(A_path)
        b = np.load(b_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load matrix files: {e}")
    
    # Validate dimensions
    assert A.shape == (d, d), f"A matrix has incorrect shape {A.shape}, expected ({d}, {d})"
    assert b.shape == (d,), f"b vector has incorrect shape {b.shape}, expected ({d},)"
    
    print(f"Successfully loaded matrices from S3: {s3_path}")
    print(f"A shape: {A.shape}, b shape: {b.shape}")
    
    return A, b