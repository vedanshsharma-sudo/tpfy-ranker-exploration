import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import argparse
import os
import random
from common.utils import print_table
from tpfy.tf_dataset.sampler import (
    TpfyImpressionSampler,
    RandomSampling,
    ZeroEntMixIn,
    SportsUserMixIn,
)
from tpfy.tf_dataset.feature import XDeepFMFeatureExtractor
from tpfy.tf_dataset.xdeepfm import BatchDatasetBuilder
from tpfy.tf_dataset.xdeepfm import parse_row
from tpfy.tf_dataset.metadata import Metadata

from datetime import datetime

from tpfy.emr_prediction.load import import_predictor
from tpfy.tpfy_config import get_tpfy_hparams

from common.constants5 import (
    S3_TPFY_IMPR_V2_TRAIN_DATASET,
    COUNTRY_KEY,
)
from common.s3_utils import download_multiple_files
from common.time_utils import timestamp
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
import numpy as np


ONE_DAY_SECS = 86400
WATCH_LEN_THRES = 600

TRAIN_FOLDER_PATH = "tmp/xdeepfm/train"
MODEL_OUTPUT_PATH = "tmp/model"

HOLDOUT_DAYS = BatchDatasetBuilder.HOLDOUT_DAYS
dataset_columns = BatchDatasetBuilder.dataset_columns

output = XDeepFMFeatureExtractor.output
output_types = tuple([o[1] for o in output])
output_shapes = tuple([o[2] for o in output])
padding_values = tuple([o[3] for o in output])


def download_user_data(date):
    s3_path = S3_TPFY_IMPR_V2_TRAIN_DATASET % (date, "data")
    print(s3_path)
    download_multiple_files(s3_path, TRAIN_FOLDER_PATH, ".parquet", "train")


def get_data_files():
    sample_data_files = sorted(
        [
            os.path.join(TRAIN_FOLDER_PATH, filename)
            for filename in os.listdir(TRAIN_FOLDER_PATH)
            if filename.endswith("parquet")
        ]
    )

    return sample_data_files


def _coalesce(a, b):
    return b if a is None else a


def _build_eval_dataset(sampler, eval_paths, hparams, metadata, date_ts):
    import pyarrow.parquet as pq

    def generator():
        paths = eval_paths
        count = 0
        for path in paths:
            print("reading eval data from %s, yield %s samples" % (path, count))
            raw = pq.read_table(path, columns=dataset_columns).to_pandas()
            n_cell = len(output)
            for index, row in raw.iterrows():
                sample = parse_row(row, metadata, hparams, date_ts)
                valid = sample is not None

                if not valid:
                    continue

                sample_data = list(sampler.generate(sample))
                batch_point = []
                if len(sample_data) == 0:
                    continue
                for i in range(n_cell):
                    max_len = max(len(point[i]) for point in sample_data)
                    batch_cell = []
                    pad = [output[i][3]]
                    for point in sample_data:
                        batch_cell.append(point[i] + pad * (max_len - len(point[i])))
                    batch_point.append(batch_cell)
                count += 1
                yield tuple(batch_point)

    return generator()


def evaluate(export_path, eval_data, hparams, eval_count=50000):
    all_predictions = []
    all_labels = []
    all_aps = []
    all_ndcgs = []
    sess, eval_output, place_holders = import_predictor(export_path)

    count = 0
    for eval_mini_batch in eval_data:
        labels = None
        for c, o in zip(eval_mini_batch, BatchDatasetBuilder.output):
            if o[0] == "target_label":
                labels = [l[0] for l in c]
        feed_dict = {pl: v for pl, v in zip(place_holders, eval_mini_batch)}
        scores = sess.run([eval_output], feed_dict=feed_dict)
        predictions = [s[0] for s in scores[0]]

        sum_labels = sum(labels)
        if sum_labels < 1:
            print("err no positive")
        elif sum_labels > len(labels) - 1:
            print("err no negative")
        else:
            all_labels.extend(labels)
            all_predictions.extend(predictions)

            ap = average_precision_score(labels, predictions)
            all_aps.append(ap)
            ndcg = ndcg_score([labels], [predictions])
            all_ndcgs.append(ndcg)
        count += 1
        if count % 1000 == 0:
            print("Evaluation consumed %s" % (count))
        if count > eval_count:
            break

    avg_ndcg = np.average(all_ndcgs)

    auc = roc_auc_score(all_labels, all_predictions)
    mean_ap = np.average(all_aps)
    return avg_ndcg, auc, mean_ap


class Evaluator:
    def __init__(
        self,
        hparams,
        model_path,
        data_paths,
        metadata,
        sample_start,
        sample_end,
        date_ts,
    ):
        self.hparams = hparams
        self.model_path = model_path
        self.data_paths = data_paths
        self.metadata = metadata
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.date_ts = date_ts

        self.results = {}
        self.common_params = dict(drop_rate=0.1, drop_days=2)

    def _do_evaluate(self, name, dataset, eval_count=50000):
        eval_start = datetime.utcnow().timestamp()
        avg_ndcg, auc, mean_ap = evaluate(
            self.model_path, dataset, self.hparams, eval_count
        )
        eval_interval = datetime.utcnow().timestamp() - eval_start
        self.results[(name, "time")] = eval_interval
        self.results[(name, "NDCG")] = avg_ndcg
        self.results[(name, "AUC")] = auc
        self.results[(name, "MAP")] = mean_ap
        print(
            "Final Eval %s  Avg NDCG %.6f, AUC: %.6f, MAP: %.6f Eval Time: %s"
            % (name, avg_ndcg, auc, mean_ap, eval_interval)
        )

    def evaluate_random(self, name, spec_mixins, eval_count=50000):
        class SamplerClass(*spec_mixins, RandomSampling):
            pass

        sampler = SamplerClass(
            self.hparams.train_neg_samples,
            self.metadata,
            XDeepFMFeatureExtractor,
            WATCH_LEN_THRES,
            self.sample_start,
            self.sample_end,
            **self.common_params,
        )
        dataset = _build_eval_dataset(
            sampler, self.data_paths, self.hparams, self.metadata, self.date_ts
        )
        self._do_evaluate(name, dataset, eval_count)
        return sampler

    def evaluate_impr(self, name, spec_mixins, eval_count=50000):
        class SamplerClass(*spec_mixins, TpfyImpressionSampler):
            pass

        sampler = SamplerClass(
            self.metadata,
            XDeepFMFeatureExtractor,
            self.sample_start,
            self.sample_end,
            **self.common_params,
        )
        dataset = _build_eval_dataset(
            sampler, self.data_paths, self.hparams, self.metadata, self.date_ts
        )
        self._do_evaluate(name, dataset, eval_count)
        return sampler

    def pretty(self):
        print_table(self.results, precision=4)


def run(args):
    random.seed(123)
    hparams = get_tpfy_hparams()
    hparams.holdout = args.holdout
    print("hparams", hparams)

    export_path = "%s/export/%s" % (MODEL_OUTPUT_PATH, args.model_name)
    metadata = Metadata(args.date)
    metadata.load()

    if not args.skip_download and args.date:
        download_user_data(args.date)
    data_files = get_data_files()

    date_ts = timestamp(args.date)
    if hparams.holdout:
        data_paths = data_files
        end = date_ts + 1 * ONE_DAY_SECS
        sample_start, sample_end = end - HOLDOUT_DAYS * ONE_DAY_SECS, None
    else:
        data_paths = data_files[: hparams.eval_file_index]
        sample_start, sample_end = None, None

    evaluator = Evaluator(
        hparams, export_path, data_paths, metadata, sample_start, sample_end, date_ts
    )
    evaluator.evaluate_random("RandomSample", [], eval_count=50000)
    evaluator.evaluate_random("RandomZeroEnt", [ZeroEntMixIn], eval_count=5000)
    sampler = evaluator.evaluate_random(
        "RandomSports", [SportsUserMixIn], eval_count=3000
    )
    print(sampler.filter_stats)

    evaluator.evaluate_impr("ImprClick", [], eval_count=50000)
    evaluator.evaluate_impr("ImprZeroEnt", [ZeroEntMixIn], eval_count=5000)
    sampler = evaluator.evaluate_impr("ImprSports", [SportsUserMixIn], eval_count=3000)
    print(sampler.filter_stats)

    evaluator.pretty()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPFY offline Evaluation.")
    parser.add_argument("model_name", type=str)
    parser.add_argument("date", type=str)
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="skip downloading evaluate data from s3",
    )
    parser.add_argument("--holdout", action="store_true")

    args = parser.parse_args()

    run(args)
