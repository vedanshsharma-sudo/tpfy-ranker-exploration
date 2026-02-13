import s3fs, os
import pyarrow
import numpy as np
from common.config.utils import data_path, model_path
from tpfy.common import TpfyDataPath
from common.config import TENANT

S3_TPFY_MODEL_EXPORT = model_path(TpfyDataPath.S3_TPFY_MODEL_EXPORT, TENANT)

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