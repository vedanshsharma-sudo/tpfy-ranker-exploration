from common.constants5 import S3_TPFY_MODEL_EXPORT
import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model.tag_constants import SERVING

from tpfy.tf_dataset.xdeepfm import BatchDatasetBuilder


def get_model_path(sql_context, model_name):
    export_path_prefix = S3_TPFY_MODEL_EXPORT % model_name
    check_point = sql_context.read \
        .csv("%s/checkpoint" % export_path_prefix) \
        .rdd \
        .flatMap(lambda x: x) \
        .collect()[0]
    model_path = "%s/export/%s" % (export_path_prefix, check_point)
    return model_path, check_point


def import_predictor(model_path):
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    sess = tf.Session(graph=tf.Graph(), config=config)
    tf.saved_model.loader.load(sess, [SERVING], model_path)
    graph = sess.graph

    place_holders = []
    for name, _, _, _ in BatchDatasetBuilder.output:
        place_holders.append(graph.get_tensor_by_name("%s:0" % name))

    eval_output = graph.get_tensor_by_name("eval_output:0")
    return sess, eval_output, place_holders
