import numpy as np
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tqdm.auto import tqdm

from tools.constant import CNN2D_CONFIG
from tools.training_model.cnn2d.gen_model_helpers.build_model import BuildModel
from tools.training_model.cnn2d.gen_model_helpers.cnn_rules import CnnRules


class GenCnn2d:
    def __init__(
        self,
        input_shape_lower=CNN2D_CONFIG["input_shape_lower"],
        input_shape_upper=CNN2D_CONFIG["input_shape_upper"],
        conv_layer_num_lower=CNN2D_CONFIG["conv_layer_num_lower"],
        conv_layer_num_upper=CNN2D_CONFIG["conv_layer_num_upper"],
        filter_lower=CNN2D_CONFIG["filter_lower"],
        filter_upper=CNN2D_CONFIG["filter_upper"],
        dense_layer_num_lower=CNN2D_CONFIG["dense_layer_num_lower"],
        dense_layer_num_upper=CNN2D_CONFIG["dense_layer_num_upper"],
        dense_size_lower=CNN2D_CONFIG["dense_size_lower"],
        dense_size_upper=CNN2D_CONFIG["dense_size_upper"],
        max_pooling_prob=CNN2D_CONFIG["max_pooling_prob"],
        input_channels=CNN2D_CONFIG["input_channels"],
        paddings=CNN2D_CONFIG["paddings"],
        activation_fcts=CNN2D_CONFIG["activation_fcts"],
        optimizers=CNN2D_CONFIG["optimizers"],
        losses=CNN2D_CONFIG["losses"],
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

        self.activations = activation_fcts
        self.optimizers = optimizers
        self.losses = losses
        self.paddings = paddings

        self.activation_pick = (
            activation_fcts if activation_fcts is not None else self.activations.copy()
        )
        self.optimizer_pick = (
            optimizers if optimizers is not None else self.optimizers.copy()
        )
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
            dense_layer_num_upper=self.dense_layer_num_upper,
        )
        layer_orders = cnn_rules.gen_cnn_rule()
        input_shape = np.random.randint(self.input_shape_lower, self.input_shape_upper)
        input_channels = np.random.choice(self.input_channels, 1)[0]
        bm = BuildModel(
            DEFAULT_INPUT_SHAPE=(input_shape, input_shape, input_channels),
            filter_lower=self.filter_lower,
            filter_upper=self.filter_upper,
            paddings=self.padding_pick,
            dense_lower=self.dense_size_lower,
            dense_upper=self.dense_size_upper,
            activations=self.activation_pick,
            optimizers=self.optimizer_pick,
            losses=self.loss_pick,
        )
        kwargs_list, layer_orders, image_shape_list = bm.generateRandomModelConfigList(
            layer_orders
        )
        return (
            kwargs_list,
            layer_orders,
            (int(input_shape), int(input_shape), int(input_channels)),
        )

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
        cnn2d.compile(metrics=["accuracy"], **kwargs["Compile"])
        return cnn2d

    def generate_model_configs(self, num_model_data=1000, progress=True):
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = GenCnn2d.nothing
        for i in loop_fun(range(num_model_data)):
            kwargs_list, layer_orders, input_shape = self.generate_cnn2d_model()
            model_configs.append([kwargs_list, layer_orders, input_shape])
        return model_configs
