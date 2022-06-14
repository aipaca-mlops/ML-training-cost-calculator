from itertools import product
from random import shuffle

import keras.applications as tka
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.models import Sequential
from tqdm.auto import tqdm

from tools.constant import CNN_PRETRAINED_CONFIG
from tools.training_model.util.time_his import TimeHistoryBasic


class GenData:
    def __init__(
        self,
        batch_sizes=CNN_PRETRAINED_CONFIG["batch_sizes"],
        optimizers=CNN_PRETRAINED_CONFIG["optimizers"],
        losses=CNN_PRETRAINED_CONFIG["losses"],
        epochs=CNN_PRETRAINED_CONFIG["epochs"],
        truncate_from=CNN_PRETRAINED_CONFIG["truncate_from"],
        trials=CNN_PRETRAINED_CONFIG["trials"],
    ):
        self.batch_sizes = batch_sizes
        self.epochs = epochs
        self.truncate_from = truncate_from
        self.trials = trials
        self.optimizers = optimizers
        self.losses = losses

    @staticmethod
    def nothing(x, **kwargs):
        return x

    @staticmethod
    def get_model(model_name, classes=None, input_shape=None):
        if classes is None:
            classes = 1000
        cls_model_method = getattr(tka, model_name)
        temp_model = cls_model_method()
        input_shape_default = temp_model.get_config()["layers"][0]["config"][
            "batch_input_shape"
        ][1:]
        if input_shape is None and classes == 1000:
            model = cls_model_method()
        elif input_shape is None:
            model = cls_model_method(
                include_top=False, input_shape=input_shape_default, classes=classes
            )
            model = Sequential([model, Flatten(), Dense(1000), Dense(classes)])
        else:
            model = cls_model_method(
                include_top=False, input_shape=tuple(input_shape), classes=classes
            )
            model = Sequential([model, Flatten(), Dense(1000), Dense(classes)])
        return model

    def get_train_data(
        self, num_data, model_name, input_shape=None, output_size=1000, progress=True
    ):
        model_data = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = GenData.nothing

        # pick random combination
        shuffle(self.batch_sizes)
        shuffle(self.optimizers)
        shuffle(self.losses)
        comb = product(self.batch_sizes, self.optimizers, self.losses)
        model_configs = []
        for _ in loop_fun(range(num_data)):
            batch_size, optimizer, loss = next(comb)
            with tf.compat.v1.Session() as sess:
                gpu_devices = tf.config.experimental.list_physical_devices("GPU")
                for device in gpu_devices:
                    tf.config.experimental.set_memory_growth(device, True)

                model = GenData.get_model(
                    model_name, classes=output_size, input_shape=input_shape
                )
                model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
                input_shape = model.get_config()["layers"][0]["config"][
                    "batch_input_shape"
                ][1:]
                batch_size_data_batch = []
                batch_size_data_epoch = []
                x = np.ones((batch_size, *input_shape), dtype=np.float32)
                y = np.ones((batch_size, output_size), dtype=np.float32)
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
            sess.close()
            batch_times_truncated = batch_size_data_batch[self.truncate_from :]
            epoch_times_truncated = batch_size_data_epoch[self.truncate_from :]
            recovered_time = [
                np.median(batch_times_truncated)
            ] * self.truncate_from + batch_times_truncated

            data_point = {
                "batch_size": batch_size,
                "batch_time_ms": np.median(batch_times_truncated),
                "epoch_time_ms": np.median(epoch_times_truncated),
                "setup_time_ms": np.sum(batch_size_data_batch) - sum(recovered_time),
                "input_dim": input_shape,
            }
            model_data.append(data_point)

            model_configs.append({"optimizer": optimizer, "loss": loss})
        return model_data, model_configs
