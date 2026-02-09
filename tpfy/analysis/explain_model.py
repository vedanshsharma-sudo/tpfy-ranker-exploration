import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import sys
import numpy as np

from common.time_utils import timestamp
from tpfy.analysis.explaination.log_utils import (
    log_behaviors,
    log_results,
    log_gradients,
)
from tpfy.analysis.explaination import baseline_target as Base

from tpfy.emr_prediction.load import import_predictor
from tpfy.tf_dataset.metadata import Metadata

from tpfy.analysis.explaination.predict_utils import *

OUTPUT = XDeepFMFeatureExtractor.output


def load_user_ids():
    reader = csv.reader(open("tpfy/analysis/explaination/user_ids.csv", "r"))
    return [pid for pid, in reader]


def get_watched_tensors(sess, tensor_names):
    watched_tensors = []
    for tensor_name in tensor_names:
        tensor = sess.graph.get_tensor_by_name(tensor_name)
        if tensor is None:
            raise RuntimeError("Cannot find %s" % tensor_name)
        watched_tensors.append(tensor)
    return watched_tensors


def build_debugging_batch(
    var_names, baseline_values, watched_tensor_values, target_index
):
    debugging_batch = []
    diffs = []

    for name, bases, targets in zip(var_names, baseline_values, watched_tensor_values):
        base, target = bases[0], targets[target_index]
        diff = (target - base) / 50
        debugging_batch.append([base + diff * (i + 1) for i in range(50)])
        diffs.append(diff)
    return debugging_batch, diffs


def run_prediction(sess, place_holders, padded_batch, eval_output, watched_tensors):
    feed_dict = {pl: v for pl, v in zip(place_holders, padded_batch)}
    scores, watched_tensor_values = sess.run(
        [eval_output, watched_tensors], feed_dict=feed_dict
    )
    scores = [s[0] for s in scores]
    return scores, watched_tensor_values


def build_integrated_gradients(
    sess,
    place_holders,
    eval_output,
    baseline_input,
    baseline_values,
    watched_tensors,
    watched_tensor_values,
    target_index,
):
    debugging_batch, diffs = build_debugging_batch(
        Base.var_names, baseline_values, watched_tensor_values, target_index
    )
    feed_dict = {
        pl: np.tile(v, (50, 1)) for pl, v in zip(place_holders, baseline_input)
    }
    feed_dict.update({pl: v for pl, v in zip(watched_tensors, debugging_batch)})

    gradients = tf.gradients(
        eval_output, watched_tensors, stop_gradients=watched_tensors
    )
    debugging_scores, debugging_gradients = sess.run(
        [eval_output, gradients], feed_dict=feed_dict
    )

    integrated_gradients = []
    var_gradients = {}
    var_cell_gradients = {}
    for name, diff, gradient in zip(Base.var_names, diffs, debugging_gradients):
        integrated_gradient = diff * np.sum(gradient, axis=0)
        integrated_gradients.append(integrated_gradient)
        var_gradients[name] = np.average(integrated_gradient)
        for i, value in enumerate(integrated_gradient):
            var_cell_gradients["%s/%s" % (name, i)] = value
    return integrated_gradients, var_gradients, var_cell_gradients, debugging_scores


def run_baseline(
    sess, meta_feature, user_feature, place_holders, eval_output, watched_tensors
):
    baseline_input = Base.get_baseline_input(
        meta_feature=meta_feature, user_feature=user_feature
    )
    feed_dict = {pl: v for pl, v in zip(place_holders, baseline_input)}
    base_scores, baseline_values = sess.run(
        [eval_output, watched_tensors], feed_dict=feed_dict
    )
    return baseline_input, baseline_values, base_scores


def select_base_target(candidates, scores):
    score_index = sorted([(s, i) for i, s in enumerate(scores)], reverse=True)
    first_valid_index = -1
    for _, i in score_index:
        if candidates[i][1] > 0:
            first_valid_index = i
            break
    return score_index[0][1], first_valid_index


def run():
    model_name, date = sys.argv[1:]

    Metadata.BASE_PATH = "tmp/tpfy_debug/%s"
    metadata = Metadata(date)
    metadata.load()

    relevances = read_relevance_table(date)

    model_path = download_model_if_needed(model_name)
    sess, eval_output, place_holders = import_predictor(model_path)

    predict_ts = timestamp(date)
    retrieve_capsule = bootstrap_retrievers(metadata, relevances, predict_ts)

    print("=========== start debugging ============")
    os.makedirs("tmp/tpfy_debug/user_log", exist_ok=True)
    pid_list = load_user_ids()
    sum_var_gradients, sum_cell_gradients = {}, {}
    for pid in pid_list:
        debug_info_out = open("tmp/tpfy_debug/user_log/%s" % pid, "w")

        watches, ns_watches = fetch_behavior_from_ubs(pid, predict_ts)  # now
        log_behaviors(
            debug_info_out, "Entertainment Watches", watches, predict_ts, metadata
        )
        log_behaviors(
            debug_info_out, "Sports Watches", ns_watches, predict_ts, metadata
        )

        user_info = fetch_info_from_dynamo(pid)

        candidates, cid_sources = retrieve(
            watches, ns_watches, predict_ts, metadata, retrieve_capsule
        )
        padded_batch, meta_feature, user_feature = extract_features(
            watches, ns_watches, user_info, predict_ts, candidates, metadata
        )

        # print('=========== prediction scores ============')
        watched_tensors = get_watched_tensors(sess, Base.tensor_names)
        scores, watched_tensor_values = run_prediction(
            sess, place_holders, padded_batch, eval_output, watched_tensors
        )
        log_results(debug_info_out, candidates, scores, cid_sources, metadata)

        # print('=========== baseline scores ============')
        baseline_input, baseline_values, base_scores = run_baseline(
            sess,
            meta_feature,
            user_feature,
            place_holders,
            eval_output,
            watched_tensors,
        )
        base_index, target_index = select_base_target(candidates, scores)
        if base_index == target_index or target_index is None:
            continue
        if base_index is not None:
            baseline_input = [
                data[base_index : base_index + 1] for data in padded_batch
            ]
            baseline_values = [
                data[base_index : base_index + 1] for data in watched_tensor_values
            ]
            base_scores = [scores[base_index : base_index + 1]]

        # print('=========== integrated gradients ============')
        (
            integrated_gradients,
            var_gradients,
            var_cell_gradients,
            debugging_scores,
        ) = build_integrated_gradients(
            sess,
            place_holders,
            eval_output,
            baseline_input,
            baseline_values,
            watched_tensors,
            watched_tensor_values,
            target_index,
        )
        log_gradients(debug_info_out, "Variable Avg Gradient", var_gradients)
        log_gradients(debug_info_out, "Variable Cell Gradient", var_cell_gradients)

        print(
            base_scores[0][0],
            target_index,
            scores[target_index],
            list(np.reshape(debugging_scores, -1)),
        )
        debug_info_out.close()

        for k, v in var_gradients.items():
            sum_var_gradients[k] = sum_var_gradients.get(k, 0) + v
        for k, v in var_cell_gradients.items():
            sum_cell_gradients[k] = sum_cell_gradients.get(k, 0) + v

    log_gradients(sys.stdout, "Variable Avg Gradient", sum_var_gradients)
    log_gradients(sys.stdout, "Variable Cell Gradient", sum_cell_gradients)
    sys.stdout.flush()


if __name__ == "__main__":
    """
    To run debugging scripts, you need:
        1. Add some user ids in tpfy/analysis/explaination/user_ids.csv
        2. Update the tensor names in tpfy/analysis/explaination/baseline_XXXX.csv
        3. Custom select_base_target function to define the target to explain and the baseline.
           By default, the baseline is with the same user-side features and "empty" item-side features.
    We can achieve further customization by:
        1. Create a new tpfy/analysis/explaination/baseline_XXX.py
        2. Change to how process behaviors from Dynamo and UBS
    """
    run()
