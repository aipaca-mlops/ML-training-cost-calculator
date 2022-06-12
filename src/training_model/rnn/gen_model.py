import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from src.training_model.constant import RNN_CONFIG


class GenModel:
    def __init__(
        self,
        activation_fcts: list = RNN_CONFIG["activation_fcts"],
        optimizers: list = RNN_CONFIG["optimizers"],
        losses: list = RNN_CONFIG["losses"],
        rnn_layers_num_lower=RNN_CONFIG["rnn_layers_num_lower"],
        rnn_layers_num_upper=RNN_CONFIG["rnn_layers_num_upper"],
        rnn_layer_size_lower=RNN_CONFIG["rnn_layer_size_lower"],
        rnn_layer_size_upper=RNN_CONFIG["rnn_layer_size_upper"],
        rnn_layer_types=RNN_CONFIG["rnn_layer_types"],
        dense_layers_num_lower=RNN_CONFIG["dense_layers_num_lower"],
        dense_layers_num_upper=RNN_CONFIG["dense_layers_num_upper"],
        dense_layer_size_lower=RNN_CONFIG["dense_layer_size_lower"],
        dense_layer_size_upper=RNN_CONFIG["dense_layer_size_upper"],
        activation_pick=RNN_CONFIG["activation_pick"],
        optimizer_pick=RNN_CONFIG["optimizer_pick"],
        loss_pick=RNN_CONFIG["loss_pick"],
        rnn_layer_type_pick=RNN_CONFIG["rnn_layer_type_pick"],
    ):
        self.rnn_layers_num_lower = rnn_layers_num_lower
        self.rnn_layers_num_upper = rnn_layers_num_upper
        self.rnn_layer_size_lower = rnn_layer_size_lower
        self.rnn_layer_size_upper = rnn_layer_size_upper
        self.dense_layers_num_lower = (dense_layers_num_lower,)
        self.dense_layers_num_upper = (dense_layers_num_upper,)
        self.dense_layer_size_lower = (dense_layer_size_lower,)
        self.dense_layer_size_upper = (dense_layer_size_upper,)
        self.activation_pick = activation_pick
        self.optimizer_pick = optimizer_pick
        self.loss_pick = loss_pick
        self.rnn_layer_type_pick = rnn_layer_type_pick
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses
        self.rnn_layer_types = rnn_layer_types

    @staticmethod
    def nothing(x):
        return x

    @staticmethod
    def build_RNN_model(
        layer_type, rnn_layer_sizes, dense_layer_sizes, activations, optimizer, loss
    ):
        if layer_type == "SimpleRNN":
            rnn_layer = SimpleRNN
        if layer_type == "LSTM":
            rnn_layer = LSTM
        if layer_type == "GRU":
            rnn_layer = GRU

        model = Sequential()
        for index, size in enumerate(rnn_layer_sizes + dense_layer_sizes):
            if index < len(rnn_layer_sizes) - 1:
                model.add(
                    rnn_layer(
                        units=size, activation=activations[index], return_sequences=True
                    )
                )
            elif index == len(rnn_layer_sizes) - 1:
                model.add(rnn_layer(units=size, activation=activations[index]))
            else:
                model.add(Dense(units=size, activation=activations[index]))
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        return model

    @staticmethod
    def get_RNN_model_features(keras_model):
        layers = [
            layer_info
            for layer_info in keras_model.get_config()["layers"]
            if layer_info["class_name"] == "SimpleRNN"
            or layer_info["class_name"] == "LSTM"
            or layer_info["class_name"] == "GRU"
            or layer_info["class_name"] == "Dense"
        ]
        layer_sizes = [lyr["config"]["units"] for lyr in layers]
        acts = [lyr["config"]["activation"].lower() for lyr in layers]
        layer_Type = [lyr["class_name"] for lyr in layers]
        return layer_sizes, acts, layer_Type

    def generate_model(self):
        rnn_layers_num = np.random.randint(
            self.rnn_layers_num_lower, self.rnn_layers_num_upper
        )
        rnn_layer_sizes = np.random.randint(
            self.rnn_layer_size_lower, self.rnn_layer_size_upper, rnn_layers_num
        )
        dense_layers_num = np.random.randint(
            self.dense_layers_num_lower, self.dense_layers_num_upper
        )
        dense_layer_sizes = np.random.randint(
            self.dense_layer_size_lower, self.dense_layer_size_upper, dense_layers_num
        )

        if self.activation_pick == "random":
            activations = np.random.choice(
                self.activation_fcts, rnn_layers_num + dense_layers_num
            )
        else:
            activations = np.random.choice(
                [self.activation_pick], rnn_layers_num + dense_layers_num
            )
        if self.optimizer_pick == "random":
            optimizer = np.random.choice(self.optimizers)
        else:
            optimizer = self.optimizer_pick
        if self.loss_pick == "random":
            loss = np.random.choice(self.losses)
        else:
            loss = self.loss_pick
        if self.rnn_layer_type_pick == "random":
            rnn_layer = np.random.choice(self.rnn_layer_types)
        else:
            rnn_layer = self.rnn_layer_type_pick

        return {
            "rnn_layer_sizes": [int(i) for i in rnn_layer_sizes],
            "dense_layer_sizes": [int(i) for i in dense_layer_sizes],
            "activations": list(activations),
            "optimizer": optimizer,
            "loss": loss,
            "rnn_type": rnn_layer,
        }

    def generate_model_configs(self, num_data=1000, progress=True):
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = GenModel.nothing
        for i in loop_fun(range(num_data)):
            data = self.generate_model()
            # del data['model']
            model_configs.append(data)
        return model_configs
