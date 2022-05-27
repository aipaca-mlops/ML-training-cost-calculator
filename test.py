"""
****************************************
 * @author: Xin Zhang
 * Date: 6/1/21
****************************************
"""
import time
import tensorflow.keras as keras
from tqdm.auto import tqdm
import numpy as np
from random import sample
import tensorflow as tf
import tensorflow.keras.applications as tka

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


class ClassicModelTrainData:
    def __init__(
            self, batch_sizes=None, optimizers=None, losses=None, epochs=None, truncate_from=None, trials=None
    ):
        self.batch_sizes = batch_sizes if batch_sizes is not None else [2 ** i for i in range(1, 9)]
        self.epochs = epochs if epochs is not None else 10
        self.truncate_from = truncate_from if truncate_from is not None else 2
        self.trials = trials if trials is not None else 5
        self.optimizers = optimizers if optimizers is not None else ["sgd", "rmsprop", "adam", "adadelta", "adagrad",
                                                                     "adamax", "nadam", "ftrl"]
        self.losses = losses if losses is not None else ["mae", "mape", "mse", "msle", "poisson",
                                                         "categorical_crossentropy"]

    @staticmethod
    def nothing(x, **kwargs):
        return x

    @staticmethod
    def get_model(model_name, classes):
        cls_model_method = getattr(tka, model_name)
        model = cls_model_method(classes=classes)
        return model

    def get_train_data(self, model_name, output_size=1000, progress=True):
        model_data = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = classic_model_train_data.nothing

        for batch_size in loop_fun(self.batch_sizes):
            for optimizer in loop_fun(self.optimizers, leave=False):
                for loss in loop_fun(self.losses, leave=False):
                    with tf.compat.v1.Session() as sess:
                        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
                        for device in gpu_devices:
                            tf.config.experimental.set_memory_growth(device, True)

                        model = classic_model_train_data.get_model(model_name, output_size)
                        model.compile(optimizer=optimizer,
                                      loss=loss,
                                      metrics=['accuracy'])
                        input_shape = model.get_config()['layers'][0]['config']['batch_input_shape'][1:]
                        batch_size_data_batch = []
                        batch_size_data_epoch = []
                        x = np.ones((batch_size, *input_shape), dtype=np.float32)
                        y = np.ones((batch_size, output_size), dtype=np.float32)
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
                    sess.close()
                    batch_times_truncated = batch_size_data_batch[self.truncate_from:]
                    epoch_times_truncated = batch_size_data_epoch[self.truncate_from:]
                    recovered_time = [
                                         np.median(batch_times_truncated)
                                     ] * self.truncate_from + batch_times_truncated

                    data_point = {'batch_size': batch_size,
                                  'optimizer': optimizer,
                                  'loss': loss,
                                  'batch_time': np.median(batch_times_truncated),
                                  'epoch_time': np.median(epoch_times_truncated),
                                  'setup_time': np.sum(batch_size_data_batch) - sum(recovered_time),
                                  'input_dim': input_shape
                                  }
                    model_data.append(data_point)
        return model_data


def demo_classic_models():
    """
    check here for all valid models: https://www.tensorflow.org/api_docs/python/tf/keras/applications
    :return:
    """
    cmtd = ClassicModelTrainData(batch_sizes=[2, 4], optimizers=["sgd", "rmsprop", "adam"],
                                    losses=["mse", "msle", "poisson", "categorical_crossentropy"])
    model_data = cmtd.get_train_data('VGG16', output_size=1000, progress=True)