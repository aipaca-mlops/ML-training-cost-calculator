"""
****************************************
 * @author: Xin Zhang
 * Date: 6/1/21
****************************************
"""
import time
import tensorflow.keras as keras
import pandas as pd
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from random import sample
from sklearn.preprocessing import MinMaxScaler
import random
import collections
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from sklearn.preprocessing import OneHotEncoder
from collections.abc import Iterable

activation_fcts = [
    'relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"
]
optimizers = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]
losses = ["mae", "mape", "mse", "msle", "poisson", "categorical_crossentropy"]
paddings = ["same", "valid"]


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_start_time = time.time()
        self.epoch_times = []
        self.batch_times = []
        self.epoch_times_detail = []
        self.batch_times_detail = []

    def on_train_end(self, logs={}):
        self.train_end_time = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time_end = time.time()
        self.epoch_times.append(epoch_time_end - self.epoch_time_start)
        self.epoch_times_detail.append((self.epoch_time_start, epoch_time_end))

    def on_train_batch_begin(self, batch, logs={}):
        self.bacth_time_start = time.time()

    def on_train_batch_end(self, batch, logs={}):
        batch_time_end = time.time()
        self.batch_times.append(batch_time_end - self.bacth_time_start)
        self.batch_times_detail.append((self.bacth_time_start, batch_time_end))

    def relative_by_train_start(self):
        self.epoch_times_detail = np.array(self.epoch_times_detail) - self.train_start_time
        self.batch_times_detail = np.array(self.batch_times_detail) - self.train_start_time
        self.train_end_time = np.array(self.train_end_time) - self.train_start_time


class ModelBuild:
    def __init__(
        self,
        DEFAULT_INPUT_SHAPE=(32, 32, 3),
        filter_lower=1,
        filter_upper=101,
        paddings=None,
        dense_lower=1,
        dense_upper=1001,
        activations=None,
        optimizers=None,
        losses=None
    ):
        self.kwargs_list: list
        self.layer_orders: list
        self.DEFAULT_INPUT_SHAPE = DEFAULT_INPUT_SHAPE
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses
        self.paddings = paddings

        OPTIONS = collections.defaultdict(dict)

        OPTIONS["Model"]["layer"] = [
            "Conv2D", "Dense", "MaxPooling2D", "Dropout", "Flatten"
        ]  # the model's layer can be either Conv2D or Dense
        OPTIONS["Compile"]["optimizer"
                           ] = optimizers if optimizers is not None else self.optimizers.copy()
        OPTIONS["Compile"]["loss"] = losses if losses is not None else self.losses.copy()
        OPTIONS["Dense"]["units"] = list(range(dense_lower, dense_upper))
        OPTIONS["Dense"]["activation"
                         ] = activations if activations is not None else self.activation_fcts.copy()
        OPTIONS["Conv2D"]["filters"] = list(range(filter_lower, filter_upper))
        OPTIONS["Conv2D"]["padding"] = paddings if paddings is not None else self.paddings.copy()
        OPTIONS["Conv2D"][
            "activation"] = activations if activations is not None else self.activation_fcts.copy()
        OPTIONS["MaxPooling2D"]["padding"
                                ] = paddings if paddings is not None else self.paddings.copy()
        OPTIONS["Dropout"]["rate"] = [0.1]

        self.options = OPTIONS

    def chooseRandomComb(self, options_layer, activations=None):
        res = dict()
        for k, v in options_layer.items():
            if k == "activation" and activations is not None:
                res[k] = random.choice(activations)
            else:
                res[k] = (random.sample(v, 1)[0])
        return res

    def generateRandomModelConfigList(self, layer_orders, input_shape=None):
        """
        Use global variable all_comb to generate random cnn model conf
        To build a model, pass the return to buildCnnModel method
        """
        if input_shape is None:
            input_shape = self.DEFAULT_INPUT_SHAPE

        def updateImageShape(_l, _kwargs, _image_shape):
            kernel_size: tuple
            if _l == "Conv2D":
                if type(_kwargs["kernel_size"]) == int:  # when kwargs["kernel_size"] was set by int
                    kernel_size = (_kwargs["kernel_size"], _kwargs["kernel_size"])
                else:
                    kernel_size = _kwargs["kernel_size"]
            elif _l == "MaxPooling2D":
                if type(_kwargs["pool_size"]) == int:  # when kwargs["kernel_size"] was set by int
                    # for program simplicity, I called pool_size as kernel_size
                    kernel_size = (_kwargs["pool_size"], _kwargs["pool_size"])
                else:
                    kernel_size = _kwargs["pool_size"]

            if type(_kwargs["strides"]) == int:  # when kwargs["strides"] was set by int
                strides = (_kwargs["strides"], _kwargs["strides"])
            else:
                strides = _kwargs["strides"]
            if _kwargs["padding"] == "valid":
                _image_shape[0] = (_image_shape[0] - kernel_size[0]) // strides[0] + 1
                _image_shape[1] = (_image_shape[1] - kernel_size[1]) // strides[1] + 1
            if _kwargs["padding"] == "same":
                if _image_shape[0] % strides[0] > 0:
                    _image_shape[0] = _image_shape[0] // strides[0] + 1
                else:
                    _image_shape[0] = _image_shape[0] // strides[0]
                if _image_shape[1] % strides[1] > 0:
                    _image_shape[1] = _image_shape[1] // strides[1] + 1
                else:
                    _image_shape[1] = _image_shape[1] // strides[1]
            assert _image_shape[0] > 0 and _image_shape[1] > 0
            return _image_shape

        def validKernelStridesSize(_l, _kwargs, _image_shape):
            if _l == "Conv2D":
                if type(_kwargs["kernel_size"]) == int:
                    kernel_size = (_kwargs["kernel_size"], _kwargs["kernel_size"])
                else:
                    kernel_size = _kwargs["kernel_size"]
            elif _l == "MaxPooling2D":
                if type(_kwargs["pool_size"]) == int:  # when kwargs["kernel_size"] was set by int
                    # for program simplicity, I called pool_size as kernel_size
                    kernel_size = (_kwargs["pool_size"], _kwargs["pool_size"])
                else:
                    kernel_size = _kwargs["pool_size"]

            if type(_kwargs["strides"]) == int:
                strides = (_kwargs["strides"], _kwargs["strides"])
            else:
                strides = _kwargs["strides"]
            judge = True
            if _l in ["Conv2D", "MaxPooling2D"]:
                judge = judge and (
                    kernel_size[0] <= _image_shape[0] and kernel_size[1] <= _image_shape[1]
                )
            judge = judge and (strides[0] <= _image_shape[0] and strides[1] <= _image_shape[1])
            if judge:
                return True
            else:
                return False

        options = self.options
        kwargs_list = []
        image_shape: list = list(input_shape[:2])
        image_shape_list: list = []
        # image_shape should end up in the same shape as model
        new_layer_orders = []
        max_strides = [3, 3]

        for i, lo in enumerate(layer_orders):
            if lo == "Dense":
                kwargs = self.chooseRandomComb(options["Dense"], options["Dense"]['activation'])
            elif lo == "Conv2D":
                if image_shape[0] == 1 or image_shape[1] == 1:
                    # if one of the image dim has only size one, we stop adding new conv2D
                    continue
                options_conv2d = options["Conv2D"].copy()
                # always ensure the kernel and strides size is smaller than the image
                options_conv2d["kernel_size"] = list(
                    zip(range(1, image_shape[0]), range(1, image_shape[1]))
                )

                options_conv2d["strides"] = [(1, 1)] * 10 + list(
                    zip(range(1, max_strides[0]), range(1, max_strides[1]))
                )
                kwargs = self.chooseRandomComb(options_conv2d)
                image_shape = updateImageShape(lo, kwargs, image_shape)
                max_strides = [
                    min(max_strides[0], max(1, image_shape[0])),
                    min(max_strides[1], max(1, image_shape[1]))
                ]
            elif lo == "MaxPooling2D":
                if image_shape[0] == 1 or image_shape[1] == 1:
                    # if one of the image dim has only size one, we stop adding new conv2D
                    continue
                options_maxpooling2d = options["MaxPooling2D"].copy()
                options_maxpooling2d["pool_size"] = list(
                    zip(range(1, image_shape[0]), range(1, image_shape[1]))
                )
                options_maxpooling2d["strides"] = [(1, 1)] * 10 + list(
                    zip(range(1, max_strides[0]), range(1, max_strides[1]))
                )
                kwargs = self.chooseRandomComb(options_maxpooling2d)
                image_shape = updateImageShape(lo, kwargs, image_shape)
                max_strides = [
                    min(max_strides[0], max(1, image_shape[0])),
                    min(max_strides[1], max(1, image_shape[1]))
                ]
            elif lo == "Dropout":
                kwargs = self.chooseRandomComb(options["Dropout"])
            elif lo == "Flatten":
                kwargs = {}
            # elif l == "AveragePooling2D":
            #   pass
            else:
                print("Error: layer order contained unsupported layer: %s" % lo)
            kwargs_list.append(kwargs)
            new_layer_orders.append(lo)
            image_shape_list.append(image_shape.copy())

        kwargs = {}
        for k in ["Compile", "Fit"]:
            kwargs[k] = {}
            for item in options[k].keys():
                kwargs[k][item] = random.sample(options[k][item], 1)[0]
        kwargs_list.append(kwargs)
        return kwargs_list, new_layer_orders, image_shape_list


class CnnRules:
    def __init__(
        self,
        conv_layer_num_lower=1,
        conv_layer_num_upper=11,
        max_pooling_prob=0.5,
        dense_layer_num_lower=1,
        dense_layer_num_upper=6
    ):
        self.conv_layer_num_lower = conv_layer_num_lower  # Rule: No Convolutional Layer After the First Dense Layer
        self.conv_layer_num_upper = conv_layer_num_upper
        self.max_pooling_prob = max_pooling_prob
        self.dense_layer_num_lower = dense_layer_num_lower
        self.dense_layer_num_upper = dense_layer_num_upper

    def gen_cnn_rule(self):
        conv_layer_num = np.random.randint(self.conv_layer_num_lower, self.conv_layer_num_upper)
        dense_layer_num = np.random.randint(self.dense_layer_num_lower, self.dense_layer_num_upper)

        rule_list = []
        for _ in range(conv_layer_num):
            rule_list.append('Conv2D')
            max_pooling_appear = np.random.choice([True, False],
                                                  size=1,
                                                  replace=True,
                                                  p=[
                                                      self.max_pooling_prob,
                                                      1 - self.max_pooling_prob
                                                  ])[0]
            if max_pooling_appear:
                rule_list.append('MaxPooling2D')

        rule_list.append('Flatten')

        rule_list.extend(['Dense'] * dense_layer_num)

        return rule_list


class gen_cnn2d:
    def __init__(
        self,
        input_shape_lower=8,
        input_shape_upper=29,
        conv_layer_num_lower=1,
        conv_layer_num_upper=51,
        filter_lower=1,
        filter_upper=101,
        dense_layer_num_lower=1,
        dense_layer_num_upper=6,
        dense_size_lower=1,
        dense_size_upper=1001,
        max_pooling_prob=.5,
        input_channels=None,
        paddings=None,
        activations=None,
        optimizers=None,
        losses=None
    ):
        self.input_shape_lower = input_shape_lower
        self.input_shape_upper = input_shape_upper
        self.input_channels = input_channels if input_channels is not None else [1, 3]
        self.conv_layer_num_lower = conv_layer_num_lower
        self.conv_layer_num_upper = conv_layer_num_upper
        self.filter_lower = filter_lower
        self.filter_upper = filter_upper
        self.dense_layer_num_lower = dense_layer_num_lower
        self.dense_layer_num_upper = dense_layer_num_upper
        self.dense_size_lower = dense_size_lower
        self.dense_size_upper = dense_size_upper
        self.max_pooling_prob = max_pooling_prob

        self.activations = [
            'relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu",
            "exponential"
        ]
        self.optimizers = [
            "sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"
        ]
        self.losses = ["mae", "mape", "mse", "msle", "poisson", "categorical_crossentropy"]
        self.paddings = ["same", "valid"]

        self.activation_pick = activations if activations is not None else self.activations.copy()
        self.optimizer_pick = optimizers if optimizers is not None else self.optimizers.copy()
        self.loss_pick = losses if losses is not None else self.losses.copy()
        self.padding_pick = paddings if paddings is not None else self.paddings.copy()

    @staticmethod
    def nothing(x):
        return x

    def generate_cnn2d_model(self):
        cnn_rules = CnnRules(
            conv_layer_num_lower=self.conv_layer_num_lower,
            conv_layer_num_upper=self.conv_layer_num_upper,
            max_pooling_prob=self.max_pooling_prob,
            dense_layer_num_lower=self.dense_layer_num_lower,
            dense_layer_num_upper=self.dense_layer_num_upper
        )
        layer_orders = cnn_rules.gen_cnn_rule()
        input_shape = np.random.randint(self.input_shape_lower, self.input_shape_upper)
        input_channels = np.random.choice(self.input_channels, 1)[0]
        mb = ModelBuild(
            DEFAULT_INPUT_SHAPE=(input_shape, input_shape, input_channels),
            filter_lower=self.filter_lower,
            filter_upper=self.filter_upper,
            paddings=self.padding_pick,
            dense_lower=self.dense_size_lower,
            dense_upper=self.dense_size_upper,
            activations=self.activation_pick,
            optimizers=self.optimizer_pick,
            losses=self.loss_pick
        )
        kwargs_list, layer_orders, image_shape_list = mb.generateRandomModelConfigList(layer_orders)
        return kwargs_list, layer_orders, (int(input_shape), int(input_shape), int(input_channels))

    @staticmethod
    def build_cnn2d_model(kwargs_list, layer_orders):
        cnn2d = Sequential()
        for i, lo in enumerate(layer_orders):
            kwargs = kwargs_list[i]
            if lo == "Dense":
                cnn2d.add(Dense(**kwargs))
            elif lo == "Conv2D":
                cnn2d.add(Conv2D(**kwargs))
            elif lo == "MaxPooling2D":
                cnn2d.add(MaxPooling2D(**kwargs))
            elif lo == "Dropout":
                cnn2d.add(Dropout(**kwargs))
            elif lo == "Flatten":
                cnn2d.add(Flatten())
        kwargs = kwargs_list[-1]
        cnn2d.compile(metrics=['accuracy'], **kwargs["Compile"])
        return cnn2d

    def generate_model_configs(self, num_model_data=1000, progress=True):
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = gen_cnn2d.nothing
        for i in loop_fun(range(num_model_data)):
            kwargs_list, layer_orders, input_shape = self.generate_cnn2d_model()
            model_configs.append([kwargs_list, layer_orders, input_shape])
        return model_configs


class cnn2d_model_train_data:
    def __init__(
        self, model_configs, batch_sizes=None, epochs=None, truncate_from=None, trials=None
    ):
        self.model_configs = []
        for info_list in model_configs:
            self.model_configs.append(info_list.copy())
        self.batch_sizes = batch_sizes if batch_sizes is not None else [2**i for i in range(1, 9)]
        self.epochs = epochs if epochs is not None else 10
        self.truncate_from = truncate_from if truncate_from is not None else 2
        self.trials = trials if trials is not None else 5
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses

    def get_train_data(self, progress=True):
        model_data = []
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = gen_cnn2d.nothing
        for info_list in self.model_configs:
            model_configs.append(info_list.copy())
        for model_config_list in loop_fun(model_configs):
            kwargs_list = model_config_list[0]
            layer_orders = model_config_list[1]
            input_shape = model_config_list[2]
            model = gen_cnn2d.build_cnn2d_model(kwargs_list, layer_orders)
            batch_size = sample(self.batch_sizes, 1)[0]
            batch_size_data_batch = []
            batch_size_data_epoch = []
            out_shape = model.get_config()['layers'][-1]['config']['units']
            x = np.ones((batch_size, *input_shape), dtype=np.float32)
            y = np.ones((batch_size, out_shape), dtype=np.float32)
            for _ in range(self.trials):
                time_callback = TimeHistory()
                model.fit(
                    x,
                    y,
                    epochs=self.epochs,
                    batch_size=batch_size,
                    callbacks=[time_callback],
                    verbose=False
                )
                times_batch = np.array(time_callback.batch_times) * 1000
                times_epoch = np.array(time_callback.epoch_times) * 1000
                batch_size_data_batch.extend(times_batch)
                batch_size_data_epoch.extend(times_epoch)

            batch_times_truncated = batch_size_data_batch[self.truncate_from:]
            epoch_times_trancuted = batch_size_data_epoch[self.truncate_from:]
            recovered_time = [
                np.median(batch_times_truncated)
            ] * self.truncate_from + batch_times_truncated

            model_config_list.append({
                'batch_size': batch_size,
                'batch_time': np.median(batch_times_truncated),
                'epoch_time': np.median(epoch_times_trancuted),
                'setup_time': np.sum(batch_size_data_batch) - sum(recovered_time),
                'input_dim': input_shape
            })
            model_data.append(model_config_list)
        return model_data

    def convert_config_data(
        self, model_data, max_layer_num=105, num_fill_na=0, name_fill_na=None, min_max_scaler=True
    ):

        feature_columns = [
            'layer_type', 'layer_size', 'kernel_size', 'strides', 'padding', 'activation',
            'optimizer', 'loss', 'batch_size', 'input_shape', 'channels'
        ]
        time_columns = ['batch_time', 'epoch_time', 'setup_time']
        feature_layer_types = ['Conv2D', 'MaxPooling2D', 'Dense']

        row_num = max([
            len(activation_fcts),
            len(optimizers),
            len(losses),
            len(paddings),
            len(feature_layer_types)
        ])
        pos_dict = dict((i, feature_columns.index(i)) for i in feature_columns)
        values_dict = {
            'activation': activation_fcts,
            'optimizer': optimizers,
            'loss': losses,
            'padding': paddings,
            'layer_type': feature_layer_types
        }
        empty_rows = [[None] * len(feature_columns)] * row_num
        empty_rows = [i[:] for i in empty_rows]  # break connection for lists
        for v_type, v_list in values_dict.items():
            for index, value in enumerate(v_list):
                empty_rows[index][pos_dict[v_type]] = value

        model_data_dfs = []
        time_rows = []
        for model_info in tqdm(model_data):
            data_rows = []
            kwargs_list = model_info[0]
            layer_orders = model_info[1]
            input_shape = model_info[2][0]
            channels = model_info[2][-1]
            train_times = model_info[3]
            for index, layer_type in enumerate(layer_orders):
                values = kwargs_list[index]
                if layer_type == 'Conv2D':
                    data_rows.append([
                        layer_type, values['filters'], values['kernel_size'][0],
                        values['strides'][0], values['padding'], values['activation'],
                        kwargs_list[-1]['Compile']['optimizer'], kwargs_list[-1]['Compile']['loss'],
                        train_times['batch_size'], input_shape, channels
                    ])
                elif layer_type == 'MaxPooling2D':
                    data_rows.append([
                        layer_type, num_fill_na, values['pool_size'][0], values['strides'][0],
                        values['padding'], name_fill_na, kwargs_list[-1]['Compile']['optimizer'],
                        kwargs_list[-1]['Compile']['loss'], train_times['batch_size'], input_shape,
                        channels
                    ])
                elif layer_type == 'Dense':
                    data_rows.append([
                        layer_type, values['units'], num_fill_na, num_fill_na, name_fill_na,
                        values['activation'], kwargs_list[-1]['Compile']['optimizer'],
                        kwargs_list[-1]['Compile']['loss'], train_times['batch_size'], input_shape,
                        channels
                    ])
                else:
                    pass
            time_rows.append([
                train_times['batch_time'], train_times['epoch_time'], train_times['setup_time']
            ])
            data_rows.extend(empty_rows)
            temp_df = pd.DataFrame(data_rows, columns=feature_columns)

            temp_df = pd.get_dummies(temp_df)
            temp_df = temp_df.drop(temp_df.index.tolist()[-len(empty_rows):])

            columns_count = len(temp_df.columns)
            zero_rows = np.zeros((max_layer_num, columns_count))
            temp_array = temp_df.to_numpy()
            temp_array = np.append(temp_array, zero_rows, 0)
            temp_array = temp_array[:max_layer_num, ]
            temp_df = pd.DataFrame(temp_array, columns=temp_df.columns)
            model_data_dfs.append(temp_df)
        time_df = pd.DataFrame(time_rows, columns=time_columns)
        if min_max_scaler:
            scaled_model_dfs = []
            scaler = MinMaxScaler()
            scaler.fit(pd.concat(model_data_dfs, axis=0).to_numpy())
            for data_df in model_data_dfs:
                scaled_data = scaler.transform(data_df.to_numpy())
                scaled_temp_df = pd.DataFrame(scaled_data, columns=temp_df.columns)
                scaled_model_dfs.append(scaled_temp_df)
            return scaled_model_dfs, time_df, scaler
        return model_data_dfs, time_df, None


class convert_cnn2d_data:
    def __init__(self):
        self.optimizers = optimizers

        unique_all_optimizers = sorted(list(set(self.optimizers)))
        enc = OneHotEncoder(handle_unknown='ignore')
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
    def get_flops_conv2d_model_config(input_shape, model_config, sum_all=True, add_pooling=True):
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
        for idx, (layer_data, layer_name) in enumerate(zip(model_config[0][:-1], model_config[1])):
            if layer_name == 'Conv2D' or layer_name == 'SeparableConv2D':
                filters = layer_data['filters']
                kernel_size = layer_data['kernel_size'][0]
                strides = layer_data['strides'][0]
                padding_method = layer_data['padding']
                previous_channels = input_shape[-1]
                if padding_method == 'same':
                    output = convert_cnn2d_data.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = convert_cnn2d_data.conv2d_layer_flops(
                        input_shape[0], input_shape[1], previous_channels, kernel_size, filters
                    )
                    conv_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, filters]
                    # conv_flops.append(np.prod(input_shape))
                else:
                    output = convert_cnn2d_data.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = convert_cnn2d_data.conv2d_layer_flops(
                        input_shape[0], input_shape[1], previous_channels, kernel_size, filters
                    )
                    conv_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, filters]
                    # conv_flops.append(np.prod(input_shape))

            if layer_name == 'MaxPooling2D' or layer_name == 'AveragePooling2D':
                kernel_size = layer_data['pool_size'][0]
                strides = layer_data['strides'][0]
                padding_method = layer_data['padding']
                previous_channels = input_shape[-1]
                if padding_method == 'same':
                    output = convert_cnn2d_data.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = convert_cnn2d_data.conv2d_layer_flops(
                        input_shape[0], input_shape[1], previous_channels, kernel_size,
                        previous_channels
                    )
                    # flops = np.prod(input_shape)
                    pool_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, previous_channels]
                else:
                    output = convert_cnn2d_data.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = convert_cnn2d_data.conv2d_layer_flops(
                        input_shape[0], input_shape[1], previous_channels, kernel_size,
                        previous_channels
                    )
                    # flops = np.prod(input_shape)
                    pool_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, previous_channels]

            if layer_name == 'ZeroPadding2D':
                w_padding_size = layer_data['padding'][0]
                h_padding_size = layer_data['padding'][1]
                input_shape = [
                    input_shape[0] + np.sum(w_padding_size),
                    input_shape[1] + np.sum(h_padding_size), input_shape[-1]
                ]
            if layer_name == 'Cropping2D':
                w_cropping_size = layer_data['cropping'][0]
                h_cropping_size = layer_data['cropping'][1]
                input_shape = [
                    input_shape[0] - np.sum(w_cropping_size),
                    input_shape[1] - np.sum(h_cropping_size), input_shape[-1]
                ]

            if layer_name == 'Dense':
                if isinstance(input_shape, Iterable):
                    input_shape = np.prod(input_shape)
                else:
                    pass
                flops = convert_cnn2d_data.dense_layer_flops(input_shape, layer_data['units'])
                input_shape = layer_data['units']
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
    def get_flops_conv2d_keras(input_shape, conv_model_obj, sum_all=True, add_pooling=True):
        conv_flops = []
        pool_flops = []
        dense_flops = []
        all_flops = []
        for idx, layer_data in enumerate(conv_model_obj.get_config()['layers']):
            layer_name = layer_data['class_name']
            layer_config = layer_data['config']
            if layer_name == 'Conv2D' or layer_name == 'SeparableConv2D':
                filters = layer_config['filters']
                kernel_size = layer_config['kernel_size'][0]
                strides = layer_config['strides'][0]
                padding_method = layer_config['padding']
                previous_channels = input_shape[-1]
                if padding_method == 'same':
                    output = convert_cnn2d_data.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = convert_cnn2d_data.conv2d_layer_flops(
                        input_shape[0], input_shape[1], previous_channels, kernel_size, filters
                    )
                    conv_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, filters]
                    # conv_flops.append(np.prod(input_shape))
                else:
                    output = convert_cnn2d_data.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = convert_cnn2d_data.conv2d_layer_flops(
                        input_shape[0], input_shape[1], previous_channels, kernel_size, filters
                    )
                    conv_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, filters]
                    # conv_flops.append(np.prod(input_shape))

            if layer_name == 'MaxPooling2D' or layer_name == 'AveragePooling2D':
                kernel_size = layer_config['pool_size'][0]
                strides = layer_config['strides'][0]
                padding_method = layer_config['padding']
                previous_channels = input_shape[-1]
                if padding_method == 'same':
                    output = convert_cnn2d_data.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = convert_cnn2d_data.conv2d_layer_flops(
                        input_shape[0], input_shape[1], previous_channels, kernel_size,
                        previous_channels
                    )
                    # flops = np.prod(input_shape)
                    pool_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, previous_channels]
                else:
                    output = convert_cnn2d_data.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    flops = convert_cnn2d_data.conv2d_layer_flops(
                        input_shape[0], input_shape[1], previous_channels, kernel_size,
                        previous_channels
                    )
                    # flops = np.prod(input_shape)
                    pool_flops.append(flops)
                    all_flops.append(flops)
                    input_shape = [output, output, previous_channels]

            if layer_name == 'ZeroPadding2D':
                w_padding_size = layer_config['padding'][0]
                h_padding_size = layer_config['padding'][1]
                input_shape = [
                    input_shape[0] + np.sum(w_padding_size),
                    input_shape[1] + np.sum(h_padding_size), input_shape[-1]
                ]
            if layer_name == 'Cropping2D':
                w_cropping_size = layer_config['cropping'][0]
                h_cropping_size = layer_config['cropping'][1]
                input_shape = [
                    input_shape[0] - np.sum(w_cropping_size),
                    input_shape[1] - np.sum(h_cropping_size), input_shape[-1]
                ]

            if layer_name == 'Dense':
                if isinstance(input_shape, Iterable):
                    input_shape = np.prod(input_shape)
                else:
                    pass
                flops = convert_cnn2d_data.dense_layer_flops(input_shape, layer_config['units'])
                input_shape = layer_config['units']
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
        for idx, (layer_data, layer_name) in enumerate(zip(model_config[0][:-1], model_config[1])):
            if layer_name == 'Conv2D' or layer_name == 'SeparableConv2D':
                filters = layer_data['filters']
                kernel_size = layer_data['kernel_size'][0]
                strides = layer_data['strides'][0]
                padding_method = layer_data['padding']
                previous_channels = input_shape[-1]
                if padding_method == 'same':
                    output = convert_cnn2d_data.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, filters]
                    conv_shape_flow.append(conv_weight * np.array(input_shape[start_from:up_to]))
                    shape_flow.append(conv_weight * np.array(input_shape[start_from:up_to]))
                    muls = kernel_size * kernel_size * previous_channels * output * output
                    multiplications.append(muls)
                else:
                    output = convert_cnn2d_data.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, filters]
                    conv_shape_flow.append(conv_weight * np.array(input_shape[start_from:up_to]))
                    shape_flow.append(conv_weight * np.array(input_shape[start_from:up_to]))
                    muls = kernel_size * kernel_size * previous_channels * output * output
                    multiplications.append(muls)
            if layer_name == 'MaxPooling2D' or layer_name == 'AveragePooling2D':
                kernel_size = layer_data['pool_size'][0]
                strides = layer_data['strides'][0]
                padding_method = layer_data['padding']
                if padding_method == 'same':
                    output = convert_cnn2d_data.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, input_shape[-1]]
                    polling_shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                    shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                else:
                    output = convert_cnn2d_data.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, input_shape[-1]]
                    polling_shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                    shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
            if layer_name == 'ZeroPadding2D':
                w_padding_size = layer_data['padding'][0]
                h_padding_size = layer_data['padding'][1]
                input_shape = [
                    input_shape[0] + np.sum(w_padding_size),
                    input_shape[1] + np.sum(h_padding_size), input_shape[-1]
                ]
                polling_shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
            if layer_name == 'Cropping2D':
                w_cropping_size = layer_data['cropping'][0]
                h_cropping_size = layer_data['cropping'][1]
                input_shape = [
                    input_shape[0] - np.sum(w_cropping_size),
                    input_shape[1] - np.sum(h_cropping_size), input_shape[-1]
                ]
                polling_shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))

            if layer_name == 'Dense':
                dense_shapes.append(layer_data['units'])
        return shape_flow, conv_shape_flow, polling_shape_flow, dense_shapes, multiplications

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
        for idx, layer_data in enumerate(conv_model_obj.get_config()['layers']):
            layer_name = layer_data['class_name']
            layer_config = layer_data['config']
            if layer_name == 'Conv2D' or layer_name == 'SeparableConv2D':
                filters = layer_config['filters']
                kernel_size = layer_config['kernel_size'][0]
                strides = layer_config['strides'][0]
                padding_method = layer_config['padding']
                previous_channels = input_shape[-1]
                if padding_method == 'same':
                    output = convert_cnn2d_data.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, filters]
                    conv_shape_flow.append(conv_weight * np.array(input_shape[start_from:up_to]))
                    shape_flow.append(conv_weight * np.array(input_shape[start_from:up_to]))
                    muls = kernel_size * kernel_size * previous_channels * output * output
                    multiplications.append(muls)
                else:
                    output = convert_cnn2d_data.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, filters]
                    conv_shape_flow.append(conv_weight * np.array(input_shape[start_from:up_to]))
                    shape_flow.append(conv_weight * np.array(input_shape[start_from:up_to]))
                    muls = kernel_size * kernel_size * previous_channels * output * output
                    multiplications.append(muls)
            if layer_name == 'MaxPooling2D' or layer_name == 'AveragePooling2D':
                kernel_size = layer_config['pool_size'][0]
                strides = layer_config['strides'][0]
                padding_method = layer_config['padding']
                if padding_method == 'same':
                    output = convert_cnn2d_data.same_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, input_shape[-1]]
                    polling_shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                    shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                else:
                    output = convert_cnn2d_data.valid_padding_output(
                        input_shape[0], kernel_size, strides
                    )
                    input_shape = [output, output, input_shape[-1]]
                    polling_shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                    shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
            if layer_name == 'ZeroPadding2D':
                w_padding_size = layer_config['padding'][0]
                h_padding_size = layer_config['padding'][1]
                input_shape = [
                    input_shape[0] + np.sum(w_padding_size),
                    input_shape[1] + np.sum(h_padding_size), input_shape[-1]
                ]
                polling_shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
            if layer_name == 'Cropping2D':
                w_cropping_size = layer_config['cropping'][0]
                h_cropping_size = layer_config['cropping'][1]
                input_shape = [
                    input_shape[0] - np.sum(w_cropping_size),
                    input_shape[1] - np.sum(h_cropping_size), input_shape[-1]
                ]
                polling_shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))

            if layer_name == 'Dense':
                dense_shapes.append(layer_config['units'])

            if layer_name == 'Functional':
                for idx, layer_data in enumerate(layer_config['layers']):
                    layer_name = layer_data['class_name']
                    layer_config = layer_data['config']
                    if layer_name == 'Conv2D' or layer_name == 'SeparableConv2D':
                        filters = layer_config['filters']
                        kernel_size = layer_config['kernel_size'][0]
                        strides = layer_config['strides'][0]
                        padding_method = layer_config['padding']
                        previous_channels = input_shape[-1]
                        if padding_method == 'same':
                            output = convert_cnn2d_data.same_padding_output(
                                input_shape[0], kernel_size, strides
                            )
                            input_shape = [output, output, filters]
                            conv_shape_flow.append(
                                conv_weight * np.array(input_shape[start_from:up_to])
                            )
                            shape_flow.append(conv_weight * np.array(input_shape[start_from:up_to]))
                            muls = kernel_size * kernel_size * previous_channels * output * output
                            multiplications.append(muls)
                        else:
                            output = convert_cnn2d_data.valid_padding_output(
                                input_shape[0], kernel_size, strides
                            )
                            input_shape = [output, output, filters]
                            conv_shape_flow.append(
                                conv_weight * np.array(input_shape[start_from:up_to])
                            )
                            shape_flow.append(conv_weight * np.array(input_shape[start_from:up_to]))
                            muls = kernel_size * kernel_size * previous_channels * output * output
                            multiplications.append(muls)
                    if layer_name == 'MaxPooling2D' or layer_name == 'AveragePooling2D':
                        kernel_size = layer_config['pool_size'][0]
                        strides = layer_config['strides'][0]
                        padding_method = layer_config['padding']
                        if padding_method == 'same':
                            output = convert_cnn2d_data.same_padding_output(
                                input_shape[0], kernel_size, strides
                            )
                            input_shape = [output, output, input_shape[-1]]
                            polling_shape_flow.append(
                                pool_weight * np.array(input_shape[start_from:up_to])
                            )
                            shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                        else:
                            output = convert_cnn2d_data.valid_padding_output(
                                input_shape[0], kernel_size, strides
                            )
                            input_shape = [output, output, input_shape[-1]]
                            polling_shape_flow.append(
                                pool_weight * np.array(input_shape[start_from:up_to])
                            )
                            shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                    if layer_name == 'ZeroPadding2D':
                        w_padding_size = layer_config['padding'][0]
                        h_padding_size = layer_config['padding'][1]
                        input_shape = [
                            input_shape[0] + np.sum(w_padding_size),
                            input_shape[1] + np.sum(h_padding_size), input_shape[-1]
                        ]
                        polling_shape_flow.append(
                            pool_weight * np.array(input_shape[start_from:up_to])
                        )
                        shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))
                    if layer_name == 'Cropping2D':
                        w_cropping_size = layer_config['cropping'][0]
                        h_cropping_size = layer_config['cropping'][1]
                        input_shape = [
                            input_shape[0] - np.sum(w_cropping_size),
                            input_shape[1] - np.sum(h_cropping_size), input_shape[-1]
                        ]
                        polling_shape_flow.append(
                            pool_weight * np.array(input_shape[start_from:up_to])
                        )
                        shape_flow.append(pool_weight * np.array(input_shape[start_from:up_to]))

                    if layer_name == 'Dense':
                        dense_shapes.append(layer_config['units'])
        return shape_flow, conv_shape_flow, polling_shape_flow, dense_shapes, multiplications

    @staticmethod
    def get_flops_tensorflow_graph2(model):
        concrete = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete.get_concrete_function([
            tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs
        ])
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
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
            tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype)
        )
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd='op', options=opts
        )
        return flops.total_float_ops

    def convert_model_config(
        self, model_config_conv2d, layer_num_upper=105, data_type='FLOPs', min_max_scaler=True
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
            batch_size = model_config[-1]['batch_size']
            optimizer = model_config[0][-1]['Compile']['optimizer']
            # conv_model = gen_cnn2d.build_cnn2d_model(model_config[0], model_config[1])
            input_shape = model_config[-2]
            # conv_model.build(input_shape=(batch_size, *input_shape))
            # flops = get_flops(conv_model, batch_size=batch_size)
            shape_flow, conv_shape_flow, polling_shape_flow, dense_shapes, multiplications = convert_cnn2d_data.get_data_shape_flow_conv2d_model_config(
                input_shape, model_config
            )
            shape_flow = [np.prod(i) for i in shape_flow]
            shape_flow = shape_flow[:layer_num_upper]
            short_position1 = layer_num_upper - len(shape_flow)
            shape_flow = shape_flow + [0] * short_position1

            flops_layer = convert_cnn2d_data.get_flops_conv2d_model_config(
                input_shape, model_config, False
            )
            flops_layer = flops_layer[:layer_num_upper]
            short_position = layer_num_upper - len(flops_layer)
            flops_layer = flops_layer + [0] * short_position

            flops_data_conv2d_layer.append(flops_layer)
            shape_flow_data.append(shape_flow)
            all_optimizers.append(optimizer)
            all_batch_sizes.append(batch_size)
            times_data_conv2d.append(model_config[-1]['batch_time'])

        conv_data = []
        if data_type.lower().startswith('f'):
            model_computation_data = flops_data_conv2d_layer.copy()
        elif data_type.lower().startswith('s'):
            model_computation_data = shape_flow_data.copy()
        else:
            model_computation_data = flops_data_conv2d_layer.copy()

        for size, batch, opt in tqdm(list(zip(model_computation_data, all_batch_sizes,
                                              all_optimizers))):
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
        data_type='FLOPs',
        scaler=None
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

        shape_flow, conv_shape_flow, polling_shape_flow, dense_shapes, multiplications = convert_cnn2d_data.get_data_shape_flow_conv2d_keras(
            input_shape, conv_model_obj
        )
        shape_flow = [np.prod(i) for i in shape_flow]
        shape_flow = shape_flow[:layer_num_upper]
        short_position1 = layer_num_upper - len(shape_flow)
        shape_flow = shape_flow + [0] * short_position1

        flops_layer = convert_cnn2d_data.get_flops_conv2d_keras(input_shape, conv_model_obj, False)
        flops_layer = flops_layer[:layer_num_upper]
        short_position = layer_num_upper - len(flops_layer)
        flops_layer = flops_layer + [0] * short_position

        if data_type.lower().startswith('f'):
            layer_data = flops_layer.copy()
        elif data_type.lower().startswith('s'):
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


def demo_depreciated():
    save_step = 100
    data_points = 10000

    split_indices = list(
        nltk.bigrams([0] + [
            v + index * save_step
            for index, v in enumerate([save_step] *
                                      (data_points // save_step) + [data_points % save_step])
        ])
    )

    gen = gen_cnn2d(
        input_shape_lower=20,
        input_shape_upper=101,
        conv_layer_num_lower=1,
        conv_layer_num_upper=51,
        filter_lower=1,
        filter_upper=101,
        dense_layer_num_lower=1,
        dense_layer_num_upper=6,
        dense_size_lower=1,
        dense_size_upper=1001,
        max_pooling_prob=.5,
        input_channels=None,
        paddings=None,
        activations=None,
        optimizers=None,
        losses=None
    )
    model_configs = gen.generate_model_configs(num_model_data=data_points, progress=True)

    # Save generated data for every 100 data points
    for start, end in tqdm(split_indices):
        model_configs_partial = model_configs[start:end]
    mtd = cnn2d_model_train_data(
        model_configs_partial, batch_sizes=None, epochs=None, truncate_from=None, trials=None
    )
    model_data = mtd.get_train_data(progress=False)

    now = datetime.datetime.now()
    file_name = f'/home/jupyter/TrainDataCurrentCNN/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}.json'
    with open(f'{file_name}', 'w') as fp:
        json.dump(model_data, fp)
    print(f'{end} data points saved!')

    # Load raw dta
    all_training_data_file = []

    for dirpath, dirnames, filenames in os.walk("TrainDataCurrentCNN"):
        for filename in [f for f in filenames if f.endswith(".json")]:
            all_training_data_file.append(os.path.join(dirpath, filename))

    model_data = []
    for name in all_training_data_file:
        with open(name, 'r') as fp:
            model_data.extend(json.load(fp))

    # 105 from conv_layer_num_upper * 2 + dense_layer_num_upper
    # * 2 because the maxpooling layer might be there

    model_data_dfs, time_df, scaler = mtd.convert_config_data(
        model_data, max_layer_num=105, num_fill_na=0, name_fill_na=None, min_max_scaler=True
    )

    x = np.array([
        data_df.to_numpy().reshape(
            model_data_dfs[0].shape[0] * model_data_dfs[0].shape[1],
        ) for data_df in model_data_dfs
    ])
    y = np.array(time_df.batch_time.tolist())

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    use_batchNormalization = False

    if use_batchNormalization:
        batch_model = Sequential()
        batch_model.add(
            Dense(2000, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu')
        )
        batch_model.add(BatchNormalization())
        batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
        batch_model.add(BatchNormalization())
        batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
        batch_model.add(BatchNormalization())
        batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
        batch_model.add(BatchNormalization())
        batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
        batch_model.add(BatchNormalization())
        batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
        batch_model.add(BatchNormalization())
        batch_model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        batch_model.compile(loss='mean_squared_error', optimizer='adam')
    else:
        batch_model = Sequential()
        batch_model.add(
            Dense(2000, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu')
        )
        batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
        batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
        batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
        batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
        batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
        batch_model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        batch_model.compile(loss='mean_squared_error', optimizer='adam')

    history_batch = batch_model.fit(
        x_train, y_train, batch_size=16, epochs=20, validation_data=(x_test, y_test), verbose=True
    )
    # summarize history for loss
    plt.plot(history_batch.history['loss'])
    plt.plot(history_batch.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    batch_y_pred = batch_model.predict(x_test)
    batch_y_pred = batch_y_pred.reshape(batch_y_pred.shape[0], )
    plt.scatter(batch_y_pred, y_test)


def demo_new():
    save_step = 100
    data_points = 10000

    split_indices = list(
        nltk.bigrams([0] + [
            v + index * save_step
            for index, v in enumerate([save_step] *
                                      (data_points // save_step) + [data_points % save_step])
        ])
    )

    gen = gen_cnn2d(
        input_shape_lower=20,
        input_shape_upper=101,
        conv_layer_num_lower=1,
        conv_layer_num_upper=51,
        filter_lower=1,
        filter_upper=101,
        dense_layer_num_lower=1,
        dense_layer_num_upper=6,
        dense_size_lower=1,
        dense_size_upper=1001,
        max_pooling_prob=.5,
        input_channels=None,
        paddings=None,
        activations=None,
        optimizers=None,
        losses=None
    )
    model_configs = gen.generate_model_configs(num_model_data=data_points, progress=True)

    # Save generated data for every 100 data points
    for start, end in tqdm(split_indices):
        model_configs_partial = model_configs[start:end]
    mtd = cnn2d_model_train_data(
        model_configs_partial, batch_sizes=None, epochs=None, truncate_from=None, trials=None
    )
    model_data = mtd.get_train_data(progress=False)

    now = datetime.datetime.now()
    file_name = f'/home/jupyter/TrainDataCurrentCNN/{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}.json'
    with open(f'{file_name}', 'w') as fp:
        json.dump(model_data, fp)
    print(f'{end} data points saved!')

    # Load raw dta
    all_training_data_file_conv2d = []

    for dirpath, dirnames, filenames in os.walk("TrainDataCurrentCNN"):
        for filename in [f for f in filenames if f.endswith(".json")]:
            all_training_data_file_conv2d.append(os.path.join(dirpath, filename))

    model_data_conv2d = []
    for name in all_training_data_file_conv2d:
        with open(name, 'r') as fp:
            model_data_conv2d.extend(json.load(fp))

    ccd = convert_cnn2d_data()

    # Convert raw data into data points
    scaler_conv_data, times_data_conv2d, scaler = ccd.convert_model_config(
        model_data_conv2d, layer_num_upper=105, data_type='FLOPs', min_max_scaler=True
    )

    import tensorflow.keras as keras
    from tensorflow.keras.models import Sequential
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, \
        BatchNormalization
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import tensorflow as tf
    from tqdm import tqdm

    def cal_score(pred, real, absolute=False):
        pred = np.array(pred).copy()
        real = np.array(real).copy()
        if absolute:
            return abs((pred - real) / real)
        else:
            return (pred - real) / real

    # train data
    x_train, x_test, y_train, y_test = train_test_split(
        scaler_conv_data, times_data_conv2d, test_size=0.1, random_state=0
    )

    batch_model = keras.Sequential()
    batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    batch_model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    batch_model.compile(loss='mean_squared_error', optimizer='adam')

    history_batch = batch_model.fit(
        x_train, y_train, batch_size=16, epochs=15, validation_data=(x_test, y_test), verbose=True
    )
    batch_y_pred = batch_model.predict(x_test)
    batch_y_pred = batch_y_pred.reshape(batch_y_pred.shape[0], )
    plt.scatter(y_test, batch_y_pred)
    plt.scatter(y_test, y_test, c='r')
    plt.title(f'{np.mean(cal_score(y_test, batch_y_pred, True))}')
    plt.show()

    # convert keras model

    import tensorflow.keras as keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, \
        BatchNormalization
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import tensorflow as tf
    from tqdm import tqdm
    np.random.seed(1000)

    # Instantiation

    def build_AlexNet(input_shape=(256, 28, 28)):

        AlexNet = Sequential()

        # 1st Convolutional Layer
        AlexNet.add(
            Conv2D(
                filters=96,
                input_shape=input_shape,
                kernel_size=(11, 11),
                strides=(4, 4),
                padding='same'
            )
        )
        AlexNet.add(Activation('relu'))
        AlexNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 2nd Convolutional Layer
        AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
        AlexNet.add(Activation('relu'))
        AlexNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 3rd Convolutional Layer
        AlexNet.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        AlexNet.add(Activation('relu'))

        # 4th Convolutional Layer
        AlexNet.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        AlexNet.add(Activation('relu'))

        # 5th Convolutional Layer
        AlexNet.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        AlexNet.add(Activation('relu'))
        AlexNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # Passing it to a Fully Connected layer
        AlexNet.add(Flatten())
        # 1st Fully Connected Layer
        AlexNet.add(Dense(4096, input_shape=(
            32,
            32,
            3,
        )))
        AlexNet.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        AlexNet.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        AlexNet.add(Dense(4096))
        AlexNet.add(Activation('relu'))
        # Add Dropout
        AlexNet.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        AlexNet.add(Dense(1000))
        AlexNet.add(Activation('relu'))
        # Add Dropout
        AlexNet.add(Dropout(0.4))

        # Output Layer
        AlexNet.add(Dense(10))
        AlexNet.add(Activation('softmax'))

        return AlexNet

    AlexNet = build_AlexNet(input_shape=(256, 28, 28))
    AlexNet_data = ccd.convert_model_keras(
        AlexNet, (256, 28, 28), 'sgd', 128, layer_num_upper=105, data_type='FLOPs', scaler=scaler
    )
