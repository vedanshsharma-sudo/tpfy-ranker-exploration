import os
from model.hparams import get_hparams, get_hparams_spark
from common.config import TENANT

HPARAMS_DIR = "./tpfy/tpfy_config/"
DEFAULT_TRAIN_YAML = f"network-{TENANT}.yaml"


def get_tpfy_hparams(fname=None):
    valid_categories = ["data", "model", "train", "info"]
    return get_hparams(HPARAMS_DIR, DEFAULT_TRAIN_YAML, valid_categories, fname)


def get_tpfy_hparams_spark(fname=None):
    valid_categories = ["data", "model", "train", "info"]
    cur_file = os.path.abspath(__file__)
    zip_end = cur_file.index("/", cur_file.index(".zip"))
    zip_path = cur_file[:zip_end]
    return get_hparams_spark(
        zip_path, "tpfy/tpfy_config/", DEFAULT_TRAIN_YAML, valid_categories, fname
    )
