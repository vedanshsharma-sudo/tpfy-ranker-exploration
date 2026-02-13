from abc import abstractmethod

import numpy as np
import tensorflow as tf2
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score

tf = tf2.compat.v1
tf.disable_v2_behavior()

from tpfy.tf_dataset.feature import schema


def log_loss(labels, logits, eps=1e-8):
    n = len(labels)
    if n == 0:
        return 0.0

    pred_probs = 1 / (1 + np.exp(-np.array(logits, dtype=np.float64)))
    pred_probs = np.clip(pred_probs, eps, 1 - eps)
    log_pos = np.log(pred_probs)
    log_neg = np.log(1 - pred_probs)

    labels = np.array(labels, dtype=np.float64)
    s = -np.sum(labels * log_pos + (1 - labels) * log_neg)
    return s / n


class UserFilter:
    def __init__(self):
        pass

    @abstractmethod
    def filter(self, user_inputs):
        pass


class FreeUserFilter(UserFilter):
    def __init__(self):
        self.free_fid = schema.raw_plan_types.hash("free")
        super().__init__()

    def filter(self, user_inputs):
        user_fids = user_inputs["user_fids"]
        for row in user_fids:
            if self.free_fid not in row:
                return False
        return True


class PaidUserFilter(UserFilter):
    def __init__(self):
        self.free_filter = FreeUserFilter()
        super().__init__()

    def filter(self, user_inputs):
        return not self.free_filter.filter(user_inputs)


class EvalMetricGroup:
    def __init__(self, name, user_filter):
        self.name = name
        suffix = "-" + name

        self.user_filter = user_filter

        self.eval_num_users = tf.placeholder(tf.int32, [])
        self.eval_num_users_summary = tf.summary.scalar(
            "Predictor/eval_num_users{}".format(suffix), self.eval_num_users
        )

        self.eval_loss = tf.placeholder(tf.float32, [])
        self.eval_loss_summary = tf.summary.scalar(
            "Predictor/eval_loss{}".format(suffix), self.eval_loss
        )

        self.eval_avg_ndcg = tf.placeholder(tf.float32, [])
        self.eval_ndcg_summary = tf.summary.scalar(
            "Predictor/eval_avg_ndcg{}".format(suffix), self.eval_avg_ndcg
        )

        self.eval_group_auc = tf.placeholder(tf.float32, [])
        self.eval_group_auc_summary = tf.summary.scalar(
            "Predictor/eval_group_auc{}".format(suffix), self.eval_group_auc
        )

        self.eval_auc = tf.placeholder(tf.float32, [])
        self.eval_auc_summary = tf.summary.scalar(
            "Predictor/eval_auc{}".format(suffix), self.eval_auc
        )

        self.eval_map = tf.placeholder(tf.float32, [])
        self.eval_map_summary = tf.summary.scalar(
            "Predictor/eval_map{}".format(suffix), self.eval_map
        )

        self.all_matched = tf.placeholder(tf.int32, [])
        self.all_matched_summary = tf.summary.scalar(
            "Predictor/eval_all_matched{}".format(suffix), self.all_matched
        )

        self.score = tf.placeholder(tf.float32, [])
        self.score_summary = tf.summary.scalar(
            "Predictor/eval_score{}".format(suffix), self.score
        )

        self.eval_summary = tf.summary.merge(
            [
                self.eval_ndcg_summary,
                self.eval_auc_summary,
                self.eval_map_summary,
                self.eval_loss_summary,
                self.eval_group_auc_summary,
            ]
        )
        self.eval_offline_summary = tf.summary.merge(
            [self.all_matched_summary, self.score_summary]
        )

        self.clear()

    def clear(self):
        self.all_watch_labels = []
        self.all_watch_predictions = []
        self.all_aps = []
        self.all_ndcgs = []
        self.all_uaucs = []
        self.num_users = 0

    def update(self, watch_labels, watch_predictions):
        self.num_users += 1
        self.all_watch_labels.extend(watch_labels)
        self.all_watch_predictions.extend(watch_predictions)

        ap = average_precision_score(watch_labels, watch_predictions)
        self.all_aps.append(ap)
        ndcg = ndcg_score([watch_labels], [watch_predictions])
        self.all_ndcgs.append(ndcg)

        pred_probs = 1 / (1 + np.exp(-np.array(watch_predictions)))
        uauc = roc_auc_score(watch_labels, pred_probs)
        self.all_uaucs.append(uauc)

    def calc_metrics(self):
        if self.num_users == 0:
            return {
                "num_users": 0,
                "loss": 0,
                "auc": 0,
                "group_auc": 0,
                "map": 0,
                "ndcg": 0,
            }
        avg_ndcg = np.average(self.all_ndcgs)

        auc = roc_auc_score(self.all_watch_labels, self.all_watch_predictions)
        mean_ap = np.average(self.all_aps)

        group_auc = np.average(self.all_uaucs)  # not weighted!
        # print("labels", self.all_watch_labels[:10])
        # print("scores", self.all_watch_predictions[:10])
        # print("probs", pred_prob[:10])
        loss = log_loss(self.all_watch_labels, self.all_watch_predictions)
        return {
            "num_users": self.num_users,
            "loss": loss,
            "auc": auc,
            "group_auc": group_auc,
            "map": mean_ap,
            "ndcg": avg_ndcg,
        }

    def build_summary(self, metrics, sess):
        summary = sess.run(
            self.eval_summary,
            feed_dict={
                self.eval_num_users: metrics["num_users"],
                self.eval_avg_ndcg: metrics["ndcg"],
                self.eval_loss: metrics["loss"],
                self.eval_auc: metrics["auc"],
                self.eval_group_auc: metrics["group_auc"],
                self.eval_map: metrics["map"],
            },
        )
        return summary
