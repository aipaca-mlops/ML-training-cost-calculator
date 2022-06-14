import copy
from random import sample

import numpy as np
from tqdm import tqdm

from tools.constant import FFNN_CONFIG
from tools.training_model.ffnn.gen_data_helpers.flop_level_data import FlopLevelData
from tools.training_model.ffnn.gen_data_helpers.model_level_data import (
    ModelLevelData,
)
from tools.training_model.ffnn.gen_model import GenModel
from tools.training_model.util.time_his import TimeHistoryBasic


class GenData(ModelLevelData, FlopLevelData):
    def __init__(
        self,
        model_configs,
        activation_fcts: list = FFNN_CONFIG["activation_fcts"],
        optimizers: list = FFNN_CONFIG["optimizers"],
        losses: list = FFNN_CONFIG["losses"],
        hidden_layers_num_lower: int = FFNN_CONFIG["hidden_layers_num_lower"],
        hidden_layers_num_upper: int = FFNN_CONFIG["hidden_layers_num_upper"],
        input_dims=list(
            range(FFNN_CONFIG["input_dim_lower"], FFNN_CONFIG["input_dim_upper"])
        ),
        batch_sizes=FFNN_CONFIG["batch_sizes"],
        epochs=FFNN_CONFIG["epochs"],
        truncate_from=FFNN_CONFIG["truncate_from"],
        trials=FFNN_CONFIG["trials"],
        batch_strategy=FFNN_CONFIG["batch_sizes_pick"],
        input_dim_strategy=FFNN_CONFIG["input_dim_strategy"],
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
        ModelLevelData.__init__(
            self,
            activation_fcts,
            optimizers,
            losses,
            hidden_layers_num_lower,
            hidden_layers_num_upper,
            model_configs,
            input_dims,
            batch_sizes,
            epochs,
            truncate_from,
            trials,
            batch_strategy,
            input_dim_strategy,
        )
        FlopLevelData.__init__(self, optimizers)

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
            model = GenModel.build_dense_model(
                layer_sizes=model_config["layer_sizes"],
                activations=model_config["activations"],
                optimizer=model_config["optimizer"],
                loss=model_config["loss"],
            )
            if self.batch_strategy == "all":
                batch_sizes = self.batch_sizes.copy()
            else:
                batch_sizes = sample(self.batch_sizes, 1)
            input_dim = sample(self.input_dims, 1)[0]
            for batch_size in batch_sizes:
                batch_size_data_batch = []
                batch_size_data_epoch = []
                if self.input_dim_strategy == "same":
                    try:
                        input_shape = model.get_config()["layers"][0]["config"]["units"]
                    except BaseException:
                        input_shape = model.get_config()["layers"][0]["config"][
                            "batch_input_shape"
                        ][1]
                else:
                    input_shape = input_dim
                out_shape = model.get_config()["layers"][-1]["config"]["units"]
                x = np.ones((batch_size, input_shape), dtype=np.float32)
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

                model_config[f"batch_size_{batch_size}"] = {
                    "batch_time_ms": np.median(batch_times_truncated),
                    "epoch_time_ms": np.median(epoch_times_trancuted),
                    "setup_time_ms": np.sum(batch_size_data_batch)
                    - sum(recovered_time),
                    "input_dim": input_dim,
                }
            model_data.append(model_config)
        return model_data
