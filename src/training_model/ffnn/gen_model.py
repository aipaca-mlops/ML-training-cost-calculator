"""
Generate Keras FFNN models
"""
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tqdm import tqdm

from src.training_model.constant import FFNN_CONFIG


class GenModel:
    def __init__(
        self,
        activation_fcts: list = FFNN_CONFIG["activation_fcts"],
        optimizers: list = FFNN_CONFIG["optimizers"],
        losses: list = FFNN_CONFIG["losses"],
        hidden_layers_num_lower: int = FFNN_CONFIG["hidden_layers_num_lower"],
        hidden_layers_num_upper: int = FFNN_CONFIG["hidden_layers_num_upper"],
        hidden_layer_size_lower: int = FFNN_CONFIG["hidden_layer_size_lower"],
        hidden_layer_size_upper: int = FFNN_CONFIG["hidden_layer_size_upper"],
        activation_pick: list = FFNN_CONFIG["activation_pick"],
        optimizer_pick: list = FFNN_CONFIG["optimizer_pick"],
        loss_pick: list = FFNN_CONFIG["loss_pick"],
    ):
        self.hidden_layers_num_lower = hidden_layers_num_lower
        self.hidden_layers_num_upper = hidden_layers_num_upper
        self.hidden_layer_size_lower = hidden_layer_size_lower
        self.hidden_layer_size_upper = hidden_layer_size_upper
        self.activation_pick = activation_pick
        self.optimizer_pick = optimizer_pick
        self.loss_pick = loss_pick
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses

    @staticmethod
    def nothing(x):
        return x

    @staticmethod
    def build_dense_model(layer_sizes, activations, optimizer, loss):
        model_dense = Sequential()
        for index, size in enumerate(layer_sizes):
            model_dense.add(Dense(size, activation=activations[index]))
        model_dense.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        return model_dense

    @staticmethod
    def get_dense_model_features(keras_model):
        layers = [
            layer_info
            for layer_info in keras_model.get_config()["layers"]
            if layer_info["class_name"] == "Dense"
        ]
        layer_sizes = [lyr["config"]["units"] for lyr in layers]
        acts = [lyr["config"]["activation"].lower() for lyr in layers]
        return layer_sizes, acts

    def generate_model(self):
        hidden_layers_num = np.random.randint(
            self.hidden_layers_num_lower, self.hidden_layers_num_upper
        )
        hidden_layer_sizes = np.random.randint(
            self.hidden_layer_size_lower,
            self.hidden_layer_size_upper,
            hidden_layers_num,
        )

        if self.activation_pick == "random":
            activations = np.random.choice(self.activation_fcts, hidden_layers_num)
        else:
            activations = np.random.choice([self.activation_pick], hidden_layers_num)
        if self.optimizer_pick == "random":
            optimizer = np.random.choice(self.optimizers)
        else:
            optimizer = self.optimizer_pick
        if self.loss_pick == "random":
            loss = np.random.choice(self.losses)
        else:
            loss = self.loss_pick

        return {
            "model": GenModel.build_dense_model(
                hidden_layer_sizes, activations, optimizer, loss
            ),
            "layer_sizes": [int(i) for i in hidden_layer_sizes],
            "activations": list(activations),
            "optimizer": optimizer,
            "loss": loss,
        }

    def generate_model_configs(self, num_model_data=1000, progress=True):
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = GenModel.nothing
        for i in loop_fun(range(num_model_data)):
            data = self.generate_model()
            del data["model"]
            model_configs.append(data)
        return model_configs
