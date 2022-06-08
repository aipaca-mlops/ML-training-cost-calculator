from collections.abc import Iterable

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)
from tqdm.auto import tqdm

from data_gen.training_model.constant import CNN_CONFIG


class FlopLevelData:
    def __init__(self, optimizers=CNN_CONFIG["optimizers"]):
        self.optimizers = optimizers

        unique_all_optimizers = sorted(list(set(self.optimizers)))
        enc = OneHotEncoder(handle_unknown="ignore")
        x_opts = [[i] for i in unique_all_optimizers]
        enc.fit(x_opts)
        self.enc = enc

    @staticmethod
    # for valid padding
    def valid_padding_output(input_size, kernel_size, stride):
        pos = kernel_size
        output = 1
        while True:
            pos += stride
            output += 1
            if pos + stride > input_size:
                break
        padding = -(input_size - pos)
        return output

    @staticmethod
    # for same padding
    def same_padding_output(input_size, kernel_size, stride):
        if stride == 1:
            return input_size
        else:
            pos = 1
            output = 1
            while True:
                pos += stride
                output += 1
                if pos + stride > input_size:
                    break
            padding = pos + kernel_size - 1 - input_size
            return output

    @staticmethod
    def conv2d_layer_flops(h, w, c, k, out):
        return h * w * (2 * c * k * k - 1) * out

    @staticmethod
    def dense_layer_flops(i, o):
        return (2 * i - 1) * o

    @staticmethod
    def get_flops_conv2d_model_config(
        input_shape, model_config, sum_all=True, add_pooling=True
    ):
        """

        @param input_shape:
        @param model_config:
        @param sum_all:
        @param add_pooling:
        @return:
        """

        conv_flops = []
        pool_flops = []
        dense_flops = []
        all_flops = []
        for idx, (layer_data, layer_name) in enumerate(
            zip(model_config[0][:-1], model_config[1])
        ):
            if layer_name == "Conv2D" or layer_name == "SeparableConv2D":
                filters = layer_data["filters"]
                kernel_size = layer_data["kernel_size"][0]
                strides = layer_data["strides"][0]
                padding_method = layer_data["padding"]
                previous_channels = input_shape[-1]
                if padding_method == "same":
                    output = FlopLevelData.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = FlopLevelData.conv2d_layer_flops(
                        input_shape[0],
                        input_shape[1],
                        previous_channels,
                        kernel_size,
                        filters,
                    )
                    conv_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, filters]
                    # conv_flops.append(np.prod(input_shape))
                else:
                    output = FlopLevelData.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = FlopLevelData.conv2d_layer_flops(
                        input_shape[0],
                        input_shape[1],
                        previous_channels,
                        kernel_size,
                        filters,
                    )
                    conv_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, filters]
                    # conv_flops.append(np.prod(input_shape))

            if layer_name == "MaxPooling2D" or layer_name == "AveragePooling2D":
                kernel_size = layer_data["pool_size"][0]
                strides = layer_data["strides"][0]
                padding_method = layer_data["padding"]
                previous_channels = input_shape[-1]
                if padding_method == "same":
                    output = FlopLevelData.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = FlopLevelData.conv2d_layer_flops(
                        input_shape[0],
                        input_shape[1],
                        previous_channels,
                        kernel_size,
                        previous_channels,
                    )
                    # flops = np.prod(input_shape)
                    pool_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, previous_channels]
                else:
                    output = FlopLevelData.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = FlopLevelData.conv2d_layer_flops(
                        input_shape[0],
                        input_shape[1],
                        previous_channels,
                        kernel_size,
                        previous_channels,
                    )
                    # flops = np.prod(input_shape)
                    pool_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, previous_channels]

            if layer_name == "ZeroPadding2D":
                w_padding_size = layer_data["padding"][0]
                h_padding_size = layer_data["padding"][1]
                input_shape = [
                    input_shape[0] + np.sum(w_padding_size),
                    input_shape[1] + np.sum(h_padding_size),
                    input_shape[-1],
                ]
            if layer_name == "Cropping2D":
                w_cropping_size = layer_data["cropping"][0]
                h_cropping_size = layer_data["cropping"][1]
                input_shape = [
                    input_shape[0] - np.sum(w_cropping_size),
                    input_shape[1] - np.sum(h_cropping_size),
                    input_shape[-1],
                ]

            if layer_name == "Dense":
                if isinstance(input_shape, Iterable):
                    input_shape = np.prod(input_shape)
                else:
                    pass
                flops = FlopLevelData.dense_layer_flops(
                    input_shape, layer_data["units"]
                )
                input_shape = layer_data["units"]
                dense_flops.append(flops)
                all_flops.append(flops)
                # dense_flops.append(input_shape)
        if sum_all:
            if add_pooling:
                return sum(all_flops)
            else:
                return sum(conv_flops + dense_flops)
        else:
            if add_pooling:
                return all_flops
            else:
                return conv_flops + dense_flops

    @staticmethod
    def get_flops_conv2d_keras(
        input_shape, conv_model_obj, sum_all=True, add_pooling=True
    ):
        conv_flops = []
        pool_flops = []
        dense_flops = []
        all_flops = []
        for idx, layer_data in enumerate(conv_model_obj.get_config()["layers"]):
            layer_name = layer_data["class_name"]
            layer_config = layer_data["config"]
            if layer_name == "Conv2D" or layer_name == "SeparableConv2D":
                filters = layer_config["filters"]
                kernel_size = layer_config["kernel_size"][0]
                strides = layer_config["strides"][0]
                padding_method = layer_config["padding"]
                previous_channels = input_shape[-1]
                if padding_method == "same":
                    output = FlopLevelData.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = FlopLevelData.conv2d_layer_flops(
                        input_shape[0],
                        input_shape[1],
                        previous_channels,
                        kernel_size,
                        filters,
                    )
                    conv_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, filters]
                    # conv_flops.append(np.prod(input_shape))
                else:
                    output = FlopLevelData.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = FlopLevelData.conv2d_layer_flops(
                        input_shape[0],
                        input_shape[1],
                        previous_channels,
                        kernel_size,
                        filters,
                    )
                    conv_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, filters]
                    # conv_flops.append(np.prod(input_shape))

            if layer_name == "MaxPooling2D" or layer_name == "AveragePooling2D":
                kernel_size = layer_config["pool_size"][0]
                strides = layer_config["strides"][0]
                padding_method = layer_config["padding"]
                previous_channels = input_shape[-1]
                if padding_method == "same":
                    output = FlopLevelData.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = FlopLevelData.conv2d_layer_flops(
                        input_shape[0],
                        input_shape[1],
                        previous_channels,
                        kernel_size,
                        previous_channels,
                    )
                    # flops = np.prod(input_shape)
                    pool_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, previous_channels]
                else:
                    output = FlopLevelData.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = FlopLevelData.conv2d_layer_flops(
                        input_shape[0],
                        input_shape[1],
                        previous_channels,
                        kernel_size,
                        previous_channels,
                    )
                    # flops = np.prod(input_shape)
                    pool_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, previous_channels]

            if layer_name == "ZeroPadding2D":
                w_padding_size = layer_config["padding"][0]
                h_padding_size = layer_config["padding"][1]
                input_shape = [
                    input_shape[0] + np.sum(w_padding_size),
                    input_shape[1] + np.sum(h_padding_size),
                    input_shape[-1],
                ]
            if layer_name == "Cropping2D":
                w_cropping_size = layer_config["cropping"][0]
                h_cropping_size = layer_config["cropping"][1]
                input_shape = [
                    input_shape[0] - np.sum(w_cropping_size),
                    input_shape[1] - np.sum(h_cropping_size),
                    input_shape[-1],
                ]

            if layer_name == "Dense":
                if isinstance(input_shape, Iterable):
                    input_shape = np.prod(input_shape)
                else:
                    pass
                flops = FlopLevelData.dense_layer_flops(
                    input_shape, layer_config["units"]
                )
                input_shape = layer_config["units"]
                dense_flops.append(flops)
                all_flops.append(flops)
                # dense_flops.append(input_shape)
        if sum_all:
            if add_pooling:
                return sum(all_flops)
            else:
                return sum(conv_flops + dense_flops)
        else:
            if add_pooling:
                return all_flops
            else:
                return conv_flops + dense_flops

    @staticmethod
    def get_data_shape_flow_conv2d_model_config(
        input_shape, model_config, start_from=1, up_to=3, conv_weight=1, pool_weight=1
    ):
        """
        Will use the image shape flow inside the conv2d model as data
        @param input_shape:
        @param model_config:
        @param start_from:
        @param up_to:
        @param conv_weight:
        @param pool_weight:
        @return:
        """

        multiplications = []
        shape_flow = []
        dense_shapes = []
        input_shape = conv_weight * np.array(input_shape[start_from:up_to])
        shape_flow.append(input_shape)
        conv_shape_flow = []
        polling_shape_flow = []
        conv_shape_flow.append(input_shape)
        for idx, (layer_data, layer_name) in enumerate(
            zip(model_config[0][:-1], model_config[1])
        ):
            if layer_name == "Conv2D" or layer_name == "SeparableConv2D":
                filters = layer_data["filters"]
                kernel_size = layer_data["kernel_size"][0]
                strides = layer_data["strides"][0]
                padding_method = layer_data["padding"]
                previous_channels = input_shape[-1]
                if padding_method == "same":
                    output = FlopLevelData.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, filters]
                    conv_shape_flow.append(
                        conv_weight * np.array(input_shape[start_from:up_to])
                    )
                    shape_flow.append(
                        conv_weight * np.array(input_shape[start_from:up_to])
                    )
                    muls = (
                        kernel_size * kernel_size * previous_channels * output * output
                    )
                    multiplications.append(muls)
                else:
                    output = FlopLevelData.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, filters]
                    conv_shape_flow.append(
                        conv_weight * np.array(input_shape[start_from:up_to])
                    )
                    shape_flow.append(
                        conv_weight * np.array(input_shape[start_from:up_to])
                    )
                    muls = (
                        kernel_size * kernel_size * previous_channels * output * output
                    )
                    multiplications.append(muls)
            if layer_name == "MaxPooling2D" or layer_name == "AveragePooling2D":
                kernel_size = layer_data["pool_size"][0]
                strides = layer_data["strides"][0]
                padding_method = layer_data["padding"]
                if padding_method == "same":
                    output = FlopLevelData.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, input_shape[-1]]
                    polling_shape_flow.append(
                        pool_weight * np.array(input_shape[start_from:up_to])
                    )
                    shape_flow.append(
                        pool_weight * np.array(input_shape[start_from:up_to])
                    )
                else:
                    output = FlopLevelData.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, input_shape[-1]]
                    polling_shape_flow.append(
                        pool_weight * np.array(input_shape[start_from:up_to])
                    )
                    shape_flow.append(
                        pool_weight * np.array(input_shape[start_from:up_to])
                    )
            if layer_name == "ZeroPadding2D":
                w_padding_size = layer_data["padding"][0]
                h_padding_size = layer_data["padding"][1]
                input_shape = [
                    input_shape[0] + np.sum(w_padding_size),
                    input_shape[1] + np.sum(h_padding_size),
                    input_shape[-1],
                ]
                polling_shape_flow.append(
                    pool_weight * np.array(input_shape[start_from:up_to])
                )
                shape_flow.append(
                    pool_weight * np.array(input_shape[start_from:up_to]))
            if layer_name == "Cropping2D":
                w_cropping_size = layer_data["cropping"][0]
                h_cropping_size = layer_data["cropping"][1]
                input_shape = [
                    input_shape[0] - np.sum(w_cropping_size),
                    input_shape[1] - np.sum(h_cropping_size),
                    input_shape[-1],
                ]
                polling_shape_flow.append(
                    pool_weight * np.array(input_shape[start_from:up_to])
                )
                shape_flow.append(
                    pool_weight * np.array(input_shape[start_from:up_to]))

            if layer_name == "Dense":
                dense_shapes.append(layer_data["units"])
        return (
            shape_flow,
            conv_shape_flow,
            polling_shape_flow,
            dense_shapes,
            multiplications,
        )

    @staticmethod
    def get_data_shape_flow_conv2d_keras(
        input_shape, conv_model_obj, start_from=1, up_to=3, conv_weight=1, pool_weight=1
    ):
        """
        Will use the image shape flow inside the conv2d model as data
        @param input_shape:
        @param conv_model_obj:
        @param start_from:
        @param up_to:
        @param conv_weight:
        @param pool_weight:
        @return:
        """
        multiplications = []
        shape_flow = []
        dense_shapes = []
        input_shape = conv_weight * np.array(input_shape[start_from:up_to])
        shape_flow.append(input_shape)
        conv_shape_flow = []
        polling_shape_flow = []
        conv_shape_flow.append(input_shape)
        for idx, layer_data in enumerate(conv_model_obj.get_config()["layers"]):
            layer_name = layer_data["class_name"]
            layer_config = layer_data["config"]
            if layer_name == "Conv2D" or layer_name == "SeparableConv2D":
                filters = layer_config["filters"]
                kernel_size = layer_config["kernel_size"][0]
                strides = layer_config["strides"][0]
                padding_method = layer_config["padding"]
                previous_channels = input_shape[-1]
                if padding_method == "same":
                    output = FlopLevelData.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, filters]
                    conv_shape_flow.append(
                        conv_weight * np.array(input_shape[start_from:up_to])
                    )
                    shape_flow.append(
                        conv_weight * np.array(input_shape[start_from:up_to])
                    )
                    muls = (
                        kernel_size * kernel_size * previous_channels * output * output
                    )
                    multiplications.append(muls)
                else:
                    output = FlopLevelData.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, filters]
                    conv_shape_flow.append(
                        conv_weight * np.array(input_shape[start_from:up_to])
                    )
                    shape_flow.append(
                        conv_weight * np.array(input_shape[start_from:up_to])
                    )
                    muls = (
                        kernel_size * kernel_size * previous_channels * output * output
                    )
                    multiplications.append(muls)
            if layer_name == "MaxPooling2D" or layer_name == "AveragePooling2D":
                kernel_size = layer_config["pool_size"][0]
                strides = layer_config["strides"][0]
                padding_method = layer_config["padding"]
                if padding_method == "same":
                    output = FlopLevelData.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, input_shape[-1]]
                    polling_shape_flow.append(
                        pool_weight * np.array(input_shape[start_from:up_to])
                    )
                    shape_flow.append(
                        pool_weight * np.array(input_shape[start_from:up_to])
                    )
                else:
                    output = FlopLevelData.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, input_shape[-1]]
                    polling_shape_flow.append(
                        pool_weight * np.array(input_shape[start_from:up_to])
                    )
                    shape_flow.append(
                        pool_weight * np.array(input_shape[start_from:up_to])
                    )
            if layer_name == "ZeroPadding2D":
                w_padding_size = layer_config["padding"][0]
                h_padding_size = layer_config["padding"][1]
                input_shape = [
                    input_shape[0] + np.sum(w_padding_size),
                    input_shape[1] + np.sum(h_padding_size),
                    input_shape[-1],
                ]
                polling_shape_flow.append(
                    pool_weight * np.array(input_shape[start_from:up_to])
                )
                shape_flow.append(
                    pool_weight * np.array(input_shape[start_from:up_to]))
            if layer_name == "Cropping2D":
                w_cropping_size = layer_config["cropping"][0]
                h_cropping_size = layer_config["cropping"][1]
                input_shape = [
                    input_shape[0] - np.sum(w_cropping_size),
                    input_shape[1] - np.sum(h_cropping_size),
                    input_shape[-1],
                ]
                polling_shape_flow.append(
                    pool_weight * np.array(input_shape[start_from:up_to])
                )
                shape_flow.append(
                    pool_weight * np.array(input_shape[start_from:up_to]))

            if layer_name == "Dense":
                dense_shapes.append(layer_config["units"])

            if layer_name == "Functional":
                for idx, layer_data in enumerate(layer_config["layers"]):
                    layer_name = layer_data["class_name"]
                    layer_config = layer_data["config"]
                    if layer_name == "Conv2D" or layer_name == "SeparableConv2D":
                        filters = layer_config["filters"]
                        kernel_size = layer_config["kernel_size"][0]
                        strides = layer_config["strides"][0]
                        padding_method = layer_config["padding"]
                        previous_channels = input_shape[-1]
                        if padding_method == "same":
                            output = FlopLevelData.same_padding_output(
                                input_shape[0], kernel_size, strides
                            )
                            input_shape = [output, output, filters]
                            conv_shape_flow.append(
                                conv_weight *
                                np.array(input_shape[start_from:up_to])
                            )
                            shape_flow.append(
                                conv_weight *
                                np.array(input_shape[start_from:up_to])
                            )
                            muls = (
                                kernel_size
                                * kernel_size
                                * previous_channels
                                * output
                                * output
                            )
                            multiplications.append(muls)
                        else:
                            output = FlopLevelData.valid_padding_output(
                                input_shape[0], kernel_size, strides
                            )
                            input_shape = [output, output, filters]
                            conv_shape_flow.append(
                                conv_weight *
                                np.array(input_shape[start_from:up_to])
                            )
                            shape_flow.append(
                                conv_weight *
                                np.array(input_shape[start_from:up_to])
                            )
                            muls = (
                                kernel_size
                                * kernel_size
                                * previous_channels
                                * output
                                * output
                            )
                            multiplications.append(muls)
                    if layer_name == "MaxPooling2D" or layer_name == "AveragePooling2D":
                        kernel_size = layer_config["pool_size"][0]
                        strides = layer_config["strides"][0]
                        padding_method = layer_config["padding"]
                        if padding_method == "same":
                            output = FlopLevelData.same_padding_output(
                                input_shape[0], kernel_size, strides
                            )
                            input_shape = [output, output, input_shape[-1]]
                            polling_shape_flow.append(
                                pool_weight *
                                np.array(input_shape[start_from:up_to])
                            )
                            shape_flow.append(
                                pool_weight *
                                np.array(input_shape[start_from:up_to])
                            )
                        else:
                            output = FlopLevelData.valid_padding_output(
                                input_shape[0], kernel_size, strides
                            )
                            input_shape = [output, output, input_shape[-1]]
                            polling_shape_flow.append(
                                pool_weight *
                                np.array(input_shape[start_from:up_to])
                            )
                            shape_flow.append(
                                pool_weight *
                                np.array(input_shape[start_from:up_to])
                            )
                    if layer_name == "ZeroPadding2D":
                        w_padding_size = layer_config["padding"][0]
                        h_padding_size = layer_config["padding"][1]
                        input_shape = [
                            input_shape[0] + np.sum(w_padding_size),
                            input_shape[1] + np.sum(h_padding_size),
                            input_shape[-1],
                        ]
                        polling_shape_flow.append(
                            pool_weight *
                            np.array(input_shape[start_from:up_to])
                        )
                        shape_flow.append(
                            pool_weight *
                            np.array(input_shape[start_from:up_to])
                        )
                    if layer_name == "Cropping2D":
                        w_cropping_size = layer_config["cropping"][0]
                        h_cropping_size = layer_config["cropping"][1]
                        input_shape = [
                            input_shape[0] - np.sum(w_cropping_size),
                            input_shape[1] - np.sum(h_cropping_size),
                            input_shape[-1],
                        ]
                        polling_shape_flow.append(
                            pool_weight *
                            np.array(input_shape[start_from:up_to])
                        )
                        shape_flow.append(
                            pool_weight *
                            np.array(input_shape[start_from:up_to])
                        )

                    if layer_name == "Dense":
                        dense_shapes.append(layer_config["units"])
        return (
            shape_flow,
            conv_shape_flow,
            polling_shape_flow,
            dense_shapes,
            multiplications,
        )

    @staticmethod
    def get_flops_tensorflow_graph2(model):
        concrete = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs]
        )
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(
            concrete_func
        )
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts
            )
            return flops.total_float_ops

    @staticmethod
    def get_flops_tensorflow_graph(model, batch_size=None):
        if batch_size is None:
            batch_size = 1

        real_model = tf.function(model).get_concrete_function(
            tf.TensorSpec(
                [batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype
            )
        )
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(
            real_model)

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="op", options=opts
        )
        return flops.total_float_ops

    def convert_model_config(
        self,
        model_config_conv2d,
        layer_num_upper=105,
        data_type="FLOPs",
        min_max_scaler=True,
    ):
        """

        @param model_config_conv2d:
        @param layer_num_upper: max number of layer data want to keep, if model layers lees than the number, padding with 0
        @param data_type: str "FLOPs" or "Shape_Flow"
        @return:
        """
        shape_flow_data = []
        flops_data_conv2d_layer = []
        times_data_conv2d = []
        all_optimizers = []
        all_batch_sizes = []
        for index, model_config in enumerate(tqdm(model_config_conv2d)):
            batch_size = model_config[-1]["batch_size"]
            optimizer = model_config[0][-1]["Compile"]["optimizer"]
            # conv_model = gen_cnn2d.build_cnn2d_model(model_config[0], model_config[1])
            input_shape = model_config[-2]
            # conv_model.build(input_shape=(batch_size, *input_shape))
            # flops = get_flops(conv_model, batch_size=batch_size)
            (
                shape_flow,
                conv_shape_flow,
                polling_shape_flow,
                dense_shapes,
                multiplications,
            ) = FlopLevelData.get_data_shape_flow_conv2d_model_config(
                input_shape, model_config
            )
            shape_flow = [np.prod(i) for i in shape_flow]
            shape_flow = shape_flow[:layer_num_upper]
            short_position1 = layer_num_upper - len(shape_flow)
            shape_flow = shape_flow + [0] * short_position1

            flops_layer = FlopLevelData.get_flops_conv2d_model_config(
                input_shape, model_config, False
            )
            flops_layer = flops_layer[:layer_num_upper]
            short_position = layer_num_upper - len(flops_layer)
            flops_layer = flops_layer + [0] * short_position

            flops_data_conv2d_layer.append(flops_layer)
            shape_flow_data.append(shape_flow)
            all_optimizers.append(optimizer)
            all_batch_sizes.append(batch_size)
            times_data_conv2d.append(model_config[-1]["batch_time"])

        conv_data = []
        if data_type.lower().startswith("f"):
            model_computation_data = flops_data_conv2d_layer.copy()
        elif data_type.lower().startswith("s"):
            model_computation_data = shape_flow_data.copy()
        else:
            model_computation_data = flops_data_conv2d_layer.copy()

        for size, batch, opt in tqdm(
            list(zip(model_computation_data, all_batch_sizes, all_optimizers))
        ):
            optimizer_onehot = list(self.enc.transform([[opt]]).toarray()[0])
            conv_data.append(size + [batch] + optimizer_onehot)

        if min_max_scaler:
            scaler = MinMaxScaler()
            scaler.fit(conv_data)
            scaler_conv_data = scaler.transform(conv_data)

            return scaler_conv_data, np.array(times_data_conv2d), scaler
        else:
            return conv_data, np.array(times_data_conv2d), None

    def convert_model_keras(
        self,
        conv_model_obj,
        input_shape,
        optimizer,
        batch_size,
        layer_num_upper=105,
        data_type="FLOPs",
        scaler=None,
    ):
        """

        @param conv_model_obj:
        @param input_shape: list of 3 int (height, width, channels)
        @param optimizer:
        @param batch_size:
        @param layer_num_upper:
        @param data_type: FLOPs or ShapeFlow
        @param scaler:  None or should from convert_model_config, but if not None, also need optimizer and batch_size not None
        @return:
        """

        (
            shape_flow,
            conv_shape_flow,
            polling_shape_flow,
            dense_shapes,
            multiplications,
        ) = FlopLevelData.get_data_shape_flow_conv2d_keras(input_shape, conv_model_obj)
        shape_flow = [np.prod(i) for i in shape_flow]
        shape_flow = shape_flow[:layer_num_upper]
        short_position1 = layer_num_upper - len(shape_flow)
        shape_flow = shape_flow + [0] * short_position1

        flops_layer = FlopLevelData.get_flops_conv2d_keras(
            input_shape, conv_model_obj, False
        )
        flops_layer = flops_layer[:layer_num_upper]
        short_position = layer_num_upper - len(flops_layer)
        flops_layer = flops_layer + [0] * short_position

        if data_type.lower().startswith("f"):
            layer_data = flops_layer.copy()
        elif data_type.lower().startswith("s"):
            layer_data = shape_flow.copy()
        else:
            layer_data = flops_layer.copy()

        optimizer_onehot = list(self.enc.transform([[optimizer]]).toarray()[0])
        layer_data = layer_data + [batch_size] + optimizer_onehot

        if scaler is not None:
            scaled_data = scaler.transform(np.array([layer_data]))
            return scaled_data
        else:
            return layer_data
