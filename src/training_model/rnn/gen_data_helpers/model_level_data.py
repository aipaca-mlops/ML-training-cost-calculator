import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from src.training_model.constant import RNN_CONFIG


class ModelLevelData:
    def __init__(
        self,
        optimizers: list = RNN_CONFIG["optimizers"],
        rnn_layer_types: list = RNN_CONFIG["rnn_layer_types"],
    ):
        self.optimizers = optimizers
        self.rnn_layer_types = rnn_layer_types

        unique_all_rnns = sorted(list(set(self.rnn_layer_types)))
        unique_all_optimizers = sorted(list(set(self.optimizers)))

        opt_enc = OneHotEncoder(handle_unknown="ignore")
        rnn_enc = OneHotEncoder(handle_unknown="ignore")

        x_rnns = [[i] for i in unique_all_rnns]
        rnn_enc.fit(x_rnns)
        x_opts = [[i] for i in unique_all_optimizers]
        opt_enc.fit(x_opts)

        self.rnn_enc = rnn_enc
        self.opt_enc = opt_enc

    @staticmethod
    def get_rnn_type(model):
        return [
            i["class_name"]
            for i in model.get_config()["layers"]
            if i["class_name"] == "SimpleRNN"
            or i["class_name"] == "LSTM"
            or i["class_name"] == "GRU"
        ][0]

    @staticmethod
    def get_units_sum_rnn_keras(dense_model_obj):
        return sum(
            [
                layer["config"]["units"]
                for layer in dense_model_obj.get_config()["layers"]
                if layer["class_name"] == "SimpleRNN"
                or layer["class_name"] == "LSTM"
                or layer["class_name"] == "GRU"
            ]
        )

    @staticmethod
    def get_units_sum_dense_keras(dense_model_obj):
        return sum(
            [
                layer["config"]["units"]
                for layer in dense_model_obj.get_config()["layers"]
                if layer["class_name"] == "Dense"
            ]
        )

    def convert_model_config(
        self, model_config_rnn, data_type="Units", min_max_scaler=True
    ):
        """

        @param model_config_dense:
        @param data_type: str "Units" or "FLOPs"
        @param min_max_scaler:
        @return:
        """
        if data_type.lower().startswith("f"):
            print("currently FLOPs is not avaliable for RNN")
        all_batch_sizes = []
        all_optimizers = []
        all_rnn_types = []
        dense_units_data = []
        rnn_units_data = []
        times_data = []
        for index, model_config in enumerate(tqdm(model_config_rnn)):
            batch_size = model_config["batch_size"]
            all_batch_sizes.append(batch_size)
            all_optimizers.append(model_config["optimizer"])
            all_rnn_types.append(model_config["rnn_type"])
            dense_units_data.append(sum(model_config["dense_layer_sizes"]))
            rnn_units_data.append(sum(model_config["rnn_layer_sizes"]))
            times_data.append(model_config["batch_time"])

        rnn_data = []
        for rnn_size, dense_size, batch, opt, rnn_type in tqdm(
            list(
                zip(
                    rnn_units_data,
                    dense_units_data,
                    all_batch_sizes,
                    all_optimizers,
                    all_rnn_types,
                )
            )
        ):
            optimizer_onehot = list(self.opt_enc.transform([[opt]]).toarray()[0])
            rnn_type_onehot = list(self.rnn_enc.transform([[rnn_type]]).toarray()[0])
            rnn_data.append(
                [rnn_size, dense_size, batch] + optimizer_onehot + rnn_type_onehot
            )

        if min_max_scaler:
            scaler = MinMaxScaler()
            scaler.fit(rnn_data)
            scaler_rnn_data = scaler.transform(rnn_data)
            return scaler_rnn_data, np.array(times_data), scaler
        else:
            return rnn_data, np.array(times_data), None

    def convert_model_keras(
        self, rnn_model_obj, optimizer, batch_size, data_type="Unit", scaler=None
    ):
        rnn_type = self.get_rnn_type(rnn_model_obj)
        dense_unit_sum = self.get_units_sum_dense_keras(rnn_model_obj)
        rnn_unit_sum = self.get_units_sum_rnn_keras(rnn_model_obj)

        optimizer_onehot = list(self.opt_enc.transform([[optimizer]]).toarray()[0])
        rnn_type_onehot = list(self.rnn_enc.transform([[rnn_type]]).toarray()[0])

        layer_data = (
            [rnn_unit_sum, dense_unit_sum, batch_size]
            + optimizer_onehot
            + rnn_type_onehot
        )

        if scaler is not None:
            scaled_data = scaler.transform(np.array([layer_data]))
            return scaled_data
        else:
            return layer_data
