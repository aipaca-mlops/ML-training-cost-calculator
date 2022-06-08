from random import sample

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from data_gen.training_model.cnn.gen_model import GenCnn2d
from data_gen.training_model.constant import CNN_CONFIG
from data_gen.training_model.constant import TRAINING_CONTEXT_MGR
from data_gen.training_model.util.time_his import TimeHistoryBasic


class Cnn2dModelTrainData:
    def __init__(
        self,
        model_configs,
        batch_sizes=None,
        epochs=None,
        truncate_from=None,
        trials=None,
        activation_fcts=CNN_CONFIG["activation_fcts"],
        optimizers=CNN_CONFIG["optimizers"],
        losses=CNN_CONFIG["losses"],
        paddings=CNN_CONFIG["paddings"],
    ):
        self.model_configs = []
        for info_list in model_configs:
            self.model_configs.append(info_list.copy())
        self.batch_sizes = (
            batch_sizes if batch_sizes is not None else [
                2 ** i for i in range(1, 9)]
        )
        self.epochs = epochs if epochs is not None else 10
        self.truncate_from = truncate_from if truncate_from is not None else 2
        self.trials = trials if trials is not None else 5
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses
        self.paddings = paddings

    def get_train_data(self, progress=True):
        model_data = []
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = GenCnn2d.nothing
        for info_list in self.model_configs:
            model_configs.append(info_list.copy())
        for model_config_list in loop_fun(model_configs):
            with tf.compat.v1.Session() as sess:
                kwargs_list = model_config_list[0]
                layer_orders = model_config_list[1]
                input_shape = model_config_list[2]
                model = GenCnn2d.build_cnn2d_model(kwargs_list, layer_orders)
                batch_size = sample(self.batch_sizes, 1)[0]
                batch_size_data_batch = []
                batch_size_data_epoch = []
                out_shape = model.get_config()["layers"][-1]["config"]["units"]
                x = np.ones((batch_size, *input_shape), dtype=np.float32)
                y = np.ones((batch_size, out_shape), dtype=np.float32)
                for _ in range(self.trials):
                    time_callback = TimeHistoryBasic()
                    with TRAINING_CONTEXT_MGR:
                        model.fit(
                            x,
                            y,
                            epochs=self.epochs,
                            batch_size=batch_size,
                            callbacks=[time_callback],
                            verbose=True,
                        )
                    times_batch = np.array(time_callback.batch_times) * 1000
                    times_epoch = np.array(time_callback.epoch_times) * 1000
                    batch_size_data_batch.extend(times_batch)
                    batch_size_data_epoch.extend(times_epoch)
            sess.close()
            batch_times_truncated = batch_size_data_batch[self.truncate_from:]
            epoch_times_truncated = batch_size_data_epoch[self.truncate_from:]
            recovered_time = [
                np.median(batch_times_truncated)
            ] * self.truncate_from + batch_times_truncated

            model_config_list.append(
                {
                    "batch_size": batch_size,
                    "batch_time": np.median(batch_times_truncated),
                    "epoch_time": np.median(epoch_times_truncated),
                    "setup_time": np.sum(batch_size_data_batch) - sum(recovered_time),
                    "input_dim": input_shape,
                }
            )
            model_data.append(model_config_list)
        return model_data

    def convert_config_data(
        self,
        model_data,
        max_layer_num=105,
        num_fill_na=0,
        name_fill_na=None,
        min_max_scaler=True,
    ):

        feature_columns = [
            "layer_type",
            "layer_size",
            "kernel_size",
            "strides",
            "padding",
            "activation",
            "optimizer",
            "loss",
            "batch_size",
            "input_shape",
            "channels",
        ]
        time_columns = ["batch_time", "epoch_time", "setup_time"]
        feature_layer_types = ["Conv2D", "MaxPooling2D", "Dense"]

        row_num = max(
            [
                len(self.activation_fcts),
                len(self.optimizers),
                len(self.losses),
                len(self.paddings),
                len(feature_layer_types),
            ]
        )
        pos_dict = dict((i, feature_columns.index(i)) for i in feature_columns)
        values_dict = {
            "activation": self.activation_fcts,
            "optimizer": self.optimizers,
            "loss": self.losses,
            "padding": self.paddings,
            "layer_type": feature_layer_types,
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
                if layer_type == "Conv2D":
                    data_rows.append(
                        [
                            layer_type,
                            values["filters"],
                            values["kernel_size"][0],
                            values["strides"][0],
                            values["padding"],
                            values["activation"],
                            kwargs_list[-1]["Compile"]["optimizer"],
                            kwargs_list[-1]["Compile"]["loss"],
                            train_times["batch_size"],
                            input_shape,
                            channels,
                        ]
                    )
                elif layer_type == "MaxPooling2D":
                    data_rows.append(
                        [
                            layer_type,
                            num_fill_na,
                            values["pool_size"][0],
                            values["strides"][0],
                            values["padding"],
                            name_fill_na,
                            kwargs_list[-1]["Compile"]["optimizer"],
                            kwargs_list[-1]["Compile"]["loss"],
                            train_times["batch_size"],
                            input_shape,
                            channels,
                        ]
                    )
                elif layer_type == "Dense":
                    data_rows.append(
                        [
                            layer_type,
                            values["units"],
                            num_fill_na,
                            num_fill_na,
                            name_fill_na,
                            values["activation"],
                            kwargs_list[-1]["Compile"]["optimizer"],
                            kwargs_list[-1]["Compile"]["loss"],
                            train_times["batch_size"],
                            input_shape,
                            channels,
                        ]
                    )
                else:
                    pass
            time_rows.append(
                [
                    train_times["batch_time"],
                    train_times["epoch_time"],
                    train_times["setup_time"],
                ]
            )
            data_rows.extend(empty_rows)
            temp_df = pd.DataFrame(data_rows, columns=feature_columns)

            temp_df = pd.get_dummies(temp_df)
            temp_df = temp_df.drop(temp_df.index.tolist()[-len(empty_rows):])

            columns_count = len(temp_df.columns)
            zero_rows = np.zeros((max_layer_num, columns_count))
            temp_array = temp_df.to_numpy()
            temp_array = np.append(temp_array, zero_rows, 0)
            temp_array = temp_array[
                :max_layer_num,
            ]
            temp_df = pd.DataFrame(temp_array, columns=temp_df.columns)
            model_data_dfs.append(temp_df)
        time_df = pd.DataFrame(time_rows, columns=time_columns)
        if min_max_scaler:
            scaled_model_dfs = []
            scaler = MinMaxScaler()
            scaler.fit(pd.concat(model_data_dfs, axis=0).to_numpy())
            for data_df in model_data_dfs:
                scaled_data = scaler.transform(data_df.to_numpy())
                scaled_temp_df = pd.DataFrame(
                    scaled_data, columns=temp_df.columns)
                scaled_model_dfs.append(scaled_temp_df)
            return scaled_model_dfs, time_df, scaler
        return model_data_dfs, time_df, None
