import os
import numpy as np

import tensorflow as tf2
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.signature_constants import PREDICT_METHOD_NAME
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def
from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.saved_model.utils import build_tensor_info
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

tf = tf2.compat.v1
from tpfy.etl.schema import TpfyDatasetSchema


def build_compact_predict_inputs():
    compact_predict_inputs = {}
    for feature in TpfyDatasetSchema.features:
        placeholder_name = feature.name + "_compact"
        shape = feature.shape
        if feature.compact_predict:
            shape = [1, *shape]
        else:
            shape = [None, *shape]
        print("predict input", feature.name, shape)
        compact_predict_inputs[feature.name] = tf.placeholder(
            dtype=feature.tf_dtype, shape=shape, name=placeholder_name
        )
    return compact_predict_inputs


def build_signature(tpfy_model, invert_task):
    compact_predict_inputs = build_compact_predict_inputs()
    compact_predict_inputs_for_model = compact_predict_inputs.copy()
    if invert_task:
        compact_predict_inputs_for_model["task"] = (
            tf.constant(1, dtype=tf.int32) - compact_predict_inputs["task"]
        )

    compact_predict_output = tpfy_model(
        compact_predict_inputs_for_model,
        training=False,
        compact=True,
    )

    output_dict = {}
    output_dict["scores"] = build_tensor_info(compact_predict_output["scores"])
    output_dict["clickwatch"] = build_tensor_info(compact_predict_output["clickwatch"])
    output_dict["lastwatch"] = build_tensor_info(compact_predict_output["lastwatch"])

    compact_input_dict = {}
    for name, placeholder in compact_predict_inputs.items():
        compact_input_dict[name] = build_tensor_info(placeholder)

    predict_signature = build_signature_def(
        inputs=compact_input_dict,
        outputs=output_dict,
        method_name=PREDICT_METHOD_NAME,
    )

    return compact_predict_inputs, compact_predict_output, predict_signature


def export_tpfy_model(
    export_path,
    sess,
    tpfy_model,
    warmup_data,
):
    tf.add_to_collection(tf.saved_model.constants.MAIN_OP_KEY, tf.tables_initializer())
    builder = SavedModelBuilder(export_path)

    predict_inputs, predict_output, predict_signature = build_signature(
        tpfy_model, invert_task=False
    )
    inv_predict_inputs, inv_predict_output, inv_predict_signature = build_signature(
        tpfy_model, invert_task=True
    )
    signature_def_map = {
        "compact_predictor": predict_signature,
        "inv_compact_predictor": inv_predict_signature,
    }

    builder.add_meta_graph_and_variables(
        sess,
        [SERVING],
        signature_def_map,
        strip_default_attrs=True,
    )

    builder.save()

    if len(warmup_data) == 0:
        return
    print("write warmup requests")
    os.makedirs(f"{export_path}/assets.extra/", exist_ok=True)
    with tf.python_io.TFRecordWriter(
        f"{export_path}/assets.extra/tf_serving_warmup_requests"
    ) as writer:
        for features, labels, meta in warmup_data[:1]:
            compact_inputs = {}
            feed_dict = {}
            for (name, placeholder) in predict_inputs.items():
                feature_values = features[name]
                input_name = name
                if placeholder.shape[0] == 1:
                    feature_values = feature_values[0:1]
                input_tensor_proto = tf.make_tensor_proto(
                    feature_values, dtype=placeholder.dtype
                )
                compact_inputs[input_name] = input_tensor_proto
                feed_dict[placeholder] = feature_values

            sess.run(predict_output, feed_dict=feed_dict)

            """
            inv_compact_inputs = {}
            inv_feed_dict = {}
            for (name, placeholder) in inv_predict_inputs.items():
                feature_values = features[name]
                input_name = name
                if placeholder.shape[0] == 1:
                    feature_values = feature_values[0:1]
                input_tensor_proto = tf.make_tensor_proto(
                    feature_values, dtype=placeholder.dtype
                )
                inv_compact_inputs[input_name] = input_tensor_proto
                inv_feed_dict[placeholder] = feature_values

            inv_values = sess.run(inv_predict_output, feed_dict=inv_feed_dict)
            print("inv values", inv_values["lastwatch"].reshape(-1)[:100])
            """

            request = predict_pb2.PredictRequest(
                inputs=compact_inputs,
                model_spec=model_pb2.ModelSpec(signature_name="compact_predictor"),
            )
            log = prediction_log_pb2.PredictionLog(
                predict_log=prediction_log_pb2.PredictLog(request=request)
            )
            writer.write(log.SerializeToString())

    export_ops = tpfy_model.export_plain_weights_ops()
    print("export weights")
    plain_weights = sess.run(export_ops)
    weights_file = os.path.join(export_path, "plain_weights.npz")
    np.savez_compressed(weights_file, **plain_weights)
    print("export weights done")
