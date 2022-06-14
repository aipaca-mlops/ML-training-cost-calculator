import copy
from random import sample

import numpy as np
from tqdm import tqdm

from tools.constant import RNN_CONFIG
from tools.training_model.rnn.gen_data_helpers.model_level_data import ModelLevelData
from tools.training_model.rnn.gen_model import GenModel
from tools.training_model.util.time_his import TimeHistoryBasic


class GenData(ModelLevelData):
    def __init__(
        self,
        model_configs,
        activation_fcts: list = RNN_CONFIG["activation_fcts"],
        optimizers: list = RNN_CONFIG["optimizers"],
        losses: list = RNN_CONFIG["losses"],
        batch_sizes=RNN_CONFIG["batch_sizes"],
        epochs=RNN_CONFIG["epochs"],
        truncate_from=RNN_CONFIG["truncate_from"],
        trials=RNN_CONFIG["trials"],
        input_dims=RNN_CONFIG["input_dims"],
        input_dim_strategy=RNN_CONFIG["input_dim_strategy"],
        rnn_layer_types: list = RNN_CONFIG["rnn_layer_types"],
    ):
        """
        @param model_configs:
        @param input_dims:  input data number of features
        @param batch_sizes:
        @param epochs:
        @param truncate_from:
        @param trials:
        @param input_dim_strategy: 'random' or 'same', same will be same size as first layer size
        """
        ModelLevelData.__init__(self, optimizers, rnn_layer_types)
        self.model_configs = []
        for info_dict in model_configs:
            d2 = copy.deepcopy(info_dict)
            self.model_configs.append(d2)
        self.input_dims = input_dims
        self.batch_sizes = batch_sizes
        self.epochs = epochs
        self.truncate_from = truncate_from
        self.trials = trials
        self.input_dim_strategy = input_dim_strategy
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses
        self.act_mapping = dict(
            (act, index + 1) for index, act in enumerate(self.activation_fcts)
        )
        self.opt_mapping = dict(
            (opt, index + 1) for index, opt in enumerate(self.optimizers)
        )
        self.loss_mapping = dict(
            (loss, index + 1) for index, loss in enumerate(self.losses)
        )

    def get_train_data(self, progress=True):
        model_data = []
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = GenModel.nothing
        for info_dict in self.model_configs:
            d2 = copy.deepcopy(info_dict)
            model_configs.append(d2)
        for model_config in loop_fun(model_configs):
            model = GenModel.build_RNN_model(
                layer_type=model_config["rnn_type"],
                rnn_layer_sizes=model_config["rnn_layer_sizes"],
                dense_layer_sizes=model_config["dense_layer_sizes"],
                activations=model_config["activations"],
                optimizer=model_config["optimizer"],
                loss=model_config["loss"],
            )
            batch_sizes = sample(self.batch_sizes, 1)
            input_dim = sample(self.input_dims, 1)[0]
            for batch_size in batch_sizes:
                batch_size_data_batch = []
                batch_size_data_epoch = []
                if self.input_dim_strategy == "same":
                    try:
                        input_shape = model.get_config()["layers"][0]["config"]["units"]
                    except Exception:
                        input_shape = model.get_config()["layers"][0]["config"][
                            "batch_input_shape"
                        ][2]
                else:
                    input_shape = input_dim
                out_shape = model.get_config()["layers"][-1]["config"]["units"]
                x = np.ones((batch_size, 1, input_shape), dtype=np.float32)
                y = np.ones((batch_size, out_shape), dtype=np.float32)
                for _ in range(self.trials):
                    time_callback = TimeHistoryBasic()
                    model.fit(
                        x,
                        y,
                        epochs=self.epochs,
                        batch_size=batch_size,
                        callbacks=[time_callback],
                        verbose=False,
                    )
                    times_batch = np.array(time_callback.batch_times) * 1000
                    times_epoch = np.array(time_callback.epoch_times) * 1000
                    batch_size_data_batch.extend(times_batch)
                    batch_size_data_epoch.extend(times_epoch)

                batch_times_truncated = batch_size_data_batch[self.truncate_from :]
                epoch_times_trancuted = batch_size_data_epoch[self.truncate_from :]
                recovered_time = [
                    np.median(batch_times_truncated)
                ] * self.truncate_from + batch_times_truncated

                model_config["batch_size"] = batch_size
                model_config["batch_time_ms"] = np.median(batch_times_truncated)
                model_config["epoch_time_ms"] = np.median(epoch_times_trancuted)
                model_config["setup_time_ms"] = np.sum(batch_size_data_batch) - sum(
                    recovered_time
                )
                model_config["input_dim"] = input_shape
            model_data.append(model_config)
        return model_data
