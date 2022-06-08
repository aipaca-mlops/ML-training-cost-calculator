import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from data_gen.training_model.ffnn.gen_model import GenModel


class FlopLevelData:
    def __init__(self, optimizers: list):
        self.optimizers = optimizers

        unique_all_optimizers = sorted(list(set(self.optimizers)))
        enc = OneHotEncoder(handle_unknown="ignore")
        x_opts = [[i] for i in unique_all_optimizers]
        enc.fit(x_opts)
        self.enc = enc

    @staticmethod
    def dense_layer_flops(i, o):
        return (2 * i - 1) * o

    @staticmethod
    def get_flops_dense(input_shape, dense_model_obj, sum_all=True):
        dense_flops = []
        for idx, layer_data in enumerate(dense_model_obj.get_config()["layers"]):
            layer_name = layer_data["class_name"]
            layer_config = layer_data["config"]
            if layer_name == "Dense":
                flops = FlopLevelData.dense_layer_flops(
                    input_shape, layer_config["units"]
                )
                input_shape = layer_config["units"]
                dense_flops.append(flops)
        if sum_all:
            return sum(dense_flops)
        else:
            return dense_flops

    @staticmethod
    def get_units_sum_dense_keras(dense_model_obj):
        return sum(
            [
                layer["config"]["units"]
                for layer in dense_model_obj.get_config()["layers"]
                if layer["class_name"] == "Dense"
            ]
        )

    def convert_config_data(
        self, model_config_dense, data_type="Units", min_max_scaler=True
    ):
        """

        @param model_config_dense:
        @param data_type: str "Units" or "FLOPs"
        @param min_max_scaler:
        @return:
        """
        all_batch_sizes = []
        all_optimizers = []
        flops_data = []
        units_data = []
        times_data = []
        for index, model_config in enumerate(tqdm(model_config_dense)):
            batch_name = [i for i in model_config.keys() if i.startswith("batch_size")][
                0
            ]
            input_shape = model_config[batch_name]["input_dim"]
            batch_size = int(batch_name.split("_")[-1])
            all_batch_sizes.append(batch_size)
            all_optimizers.append(model_config["optimizer"])
            if data_type.lower().startswith("f"):
                model = GenModel.build_dense_model(
                    layer_sizes=model_config["layer_sizes"],
                    activations=model_config["activations"],
                    optimizer=model_config["optimizer"],
                    loss=model_config["loss"],
                )

                flops = FlopLevelData.get_flops_dense(
                    input_shape, model, sum_all=True)
                flops_data.append(flops)
            units_data.append(sum(model_config["layer_sizes"]))
            times_data.append(model_config[batch_name]["batch_time"])

        if data_type.lower().startswith("u"):
            layer_data = units_data.copy()
        elif data_type.lower().startswith("f"):
            layer_data = flops_data.copy()
        else:
            layer_data = units_data.copy()

        dense_data = []
        for size, batch, opt in tqdm(
            list(zip(layer_data, all_batch_sizes, all_optimizers))
        ):
            optimizer_onehot = list(self.enc.transform([[opt]]).toarray()[0])
            dense_data.append([size] + [batch] + optimizer_onehot)

        if min_max_scaler:
            scaler = MinMaxScaler()
            scaler.fit(dense_data)
            scaler_dense_data = scaler.transform(dense_data)
            return scaler_dense_data, np.array(times_data), scaler
        else:
            return dense_data, np.array(times_data), None

    def convert_model_data(
        self,
        input_shape,
        dense_model_obj,
        optimizer,
        batch_size,
        data_type="Unit",
        scaler=None,
    ):
        flops = FlopLevelData.get_flops_dense(
            input_shape, dense_model_obj, sum_all=True
        )
        unit_sum = FlopLevelData.get_units_sum_dense_keras(dense_model_obj)

        if data_type.lower().startswith("f"):
            layer_data = flops
        elif data_type.lower().startswith("u"):
            layer_data = unit_sum
        else:
            layer_data = unit_sum

        optimizer_onehot = list(self.enc.transform([[optimizer]]).toarray()[0])
        layer_data = [layer_data] + [batch_size] + optimizer_onehot

        if scaler is not None:
            scaled_data = scaler.transform(np.array([layer_data]))
            return scaled_data
        else:
            return layer_data
