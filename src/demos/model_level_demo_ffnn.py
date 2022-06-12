import math
import random

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from tqdm import tqdm

from src.training_model.ffnn.gen_data import GenData
from src.training_model.ffnn.gen_model import GenModel
from src.training_model.util.time_his import TimeHistoryBasic


def prepare_data(data_points: int):
    gnn = GenModel()
    model_configs = gnn.generate_model_configs(num_model_data=data_points)

    # train generated model configurations to get training time
    mtd = GenData(
        model_configs,
        input_dims=list(range(1, 1001)),
        batch_sizes=[2 ** i for i in range(1, 9)],
        epochs=5,
        truncate_from=1,
        trials=2,
        batch_strategy="random",
    )
    model_data = mtd.get_train_data()

    # convert raw data as dataframe and scaler
    df, scaler = mtd.convert_config_data(
        model_data,
        layer_num_upper=50,
        layer_na_fill=0,
        act_na_fill=0,
        min_max_scaler=True,
    )
    return df, scaler


def split_data(df, test_ratio=0.2):
    df_index = df.index.tolist()
    np.random.shuffle(df_index)

    middle_index = int(df.shape[0] * test_ratio)
    test_idx = df_index[:middle_index]
    train_idx = df_index[middle_index:]

    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    # we need to train 2 models, one to predict batch runtime, one to predict setup time
    # combine both will be the true training time of a model
    feature_cols = df.columns.tolist()[:-3]
    target_col = "batch_time"
    setup_col = "setup_time"

    x_train = df_train[feature_cols].to_numpy()
    y_batch_train = np.array(df_train[target_col].tolist())
    y_setup_train = np.array(df_train[setup_col].tolist())

    x_test = df_test[feature_cols].to_numpy()
    y_batch_test = np.array(df_test[target_col].tolist())
    y_setup_test = np.array(df_test[setup_col].tolist())
    return x_train, y_batch_train, y_setup_train, x_test, y_batch_test, y_setup_test


def build_batch_model(input_dim):
    batch_model = Sequential()
    batch_model.add(
        Dense(
            200,
            input_dim=input_dim,
            kernel_initializer="normal",
            activation="relu",
        )
    )
    batch_model.add(Dense(200, kernel_initializer="normal", activation="relu"))
    batch_model.add(Dense(200, kernel_initializer="normal", activation="relu"))
    batch_model.add(Dense(200, kernel_initializer="normal", activation="relu"))
    batch_model.add(Dense(1, kernel_initializer="normal"))
    return batch_model


def train_batch_model(x_train, y_batch_train, x_test, y_batch_test, plot=False):
    batch_model = build_batch_model(x_train.shape[1])

    # Compile model
    batch_model.compile(loss="mean_squared_error", optimizer="adam")

    history_batch = batch_model.fit(
        x_train,
        y_batch_train,
        batch_size=16,
        epochs=50,
        validation_data=(x_test, y_batch_test),
        verbose=True,
    )

    if plot:
        # summarize history for loss
        plt.plot(history_batch.history["loss"])
        plt.plot(history_batch.history["val_loss"])
        plt.title("Batch Model Epoch Vs Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

        # plot predictions vs true for batch model
        batch_y_pred = batch_model.predict(x_test)
        batch_y_pred = batch_y_pred.reshape(
            batch_y_pred.shape[0],
        )
        plt.scatter(batch_y_pred, y_batch_test)
        plt.title("Batch Time Prediction Vs Test")
        plt.show()

    return batch_model


def build_setup_model(input_dim):
    setup_model = Sequential()
    setup_model.add(
        Dense(
            200,
            input_dim=input_dim,
            kernel_initializer="normal",
            activation="relu",
        )
    )
    setup_model.add(Dense(200, kernel_initializer="normal", activation="relu"))
    setup_model.add(Dense(200, kernel_initializer="normal", activation="relu"))
    setup_model.add(Dense(200, kernel_initializer="normal", activation="relu"))
    setup_model.add(Dense(1, kernel_initializer="normal"))
    return setup_model


def train_setup_model(x_train, y_setup_train, x_test, y_setup_test, plot=False):
    setup_model = build_setup_model(x_train.shape[1])
    # Compile model
    setup_model.compile(loss="mean_squared_error", optimizer="adam")
    history_setup = setup_model.fit(
        x_train,
        y_setup_train,
        batch_size=16,
        epochs=45,
        validation_data=(x_test, y_setup_test),
        verbose=True,
    )

    if plot:
        # summarize history for loss
        plt.plot(history_setup.history["loss"])
        plt.plot(history_setup.history["val_loss"])
        plt.title("Setup Model Epoch vs Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

        # plot predictions vs true for setup time model
        setup_y_pred = setup_model.predict(x_test)
        setup_y_pred = setup_y_pred.reshape(
            setup_y_pred.shape[0],
        )
        plt.scatter(setup_y_pred, y_setup_test)
        plt.title("Setup Time Prediction Vs Test")
        plt.show()
    return setup_model


def cal_score(pred, real, absolute=False):
    # define a function to calculate error
    pred = np.array(pred).copy()
    real = np.array(real).copy()
    if absolute:
        return abs((pred - real) / real)
    else:
        return (pred - real) / real


def test_tt_prediction(
    batch_model, setup_model, scaler, plot=True, val_data_points=100
):
    val_genn = GenModel()
    val_model_configs = val_genn.generate_model_configs(num_model_data=val_data_points)

    # collect all info during training
    real_time_process_first_batchs = []
    real_time_batchs = []
    real_time_epochs = []
    real_time_start_ends = []
    y_val_preds_batch = []
    y_val_preds_setup = []
    batch_sizes_collect = []
    epochs_collect = []
    data_points_collect = []

    mtd_val = GenData([])
    for m_config in tqdm(val_model_configs):
        # here we consider changeable data size and epoch
        batch_size_val = random.sample(mtd_val.batch_sizes, 1)[0]
        epochs_val = random.sample([2, 3, 4, 5], 1)[0]
        data_dim_val = random.sample(list(range(1, 1001)), 1)[0]
        data_size_val = random.sample([5000, 10000, 15000, 1000], 1)[0]
        data_points_collect.append(data_size_val)
        batch_sizes_collect.append(batch_size_val)
        epochs_collect.append(epochs_val)

        model_val = GenModel.build_dense_model(
            layer_sizes=m_config["layer_sizes"],
            activations=m_config["activations"],
            optimizer=m_config["optimizer"],
            loss=m_config["loss"],
        )

        out_shape = model_val.get_config()["layers"][-1]["config"]["units"]
        x = np.ones((data_size_val, data_dim_val), dtype=np.float32)
        y = np.ones((data_size_val, out_shape), dtype=np.float32)

        time_callback = TimeHistoryBasic()
        model_val.fit(
            x,
            y,
            epochs=epochs_val,
            batch_size=batch_size_val,
            callbacks=[time_callback],
            verbose=False,
        )

        batch_median = np.median(time_callback.batch_times[2:])
        # remove first batch to remove the effect of setup, and compensate with
        # median batch time
        real_time_process_first_batchs.append(
            sum([batch_median] + time_callback.batch_times[1:]) * 1000
        )
        real_time_batchs.append(sum(time_callback.batch_times) * 1000)
        real_time_epochs.append(sum(time_callback.epoch_times) * 1000)
        real_time_start_ends.append(
            (time_callback.train_end_time - time_callback.train_start_time) * 1000
        )

        train_batch_numbers = math.ceil(data_size_val / batch_size_val) * epochs_val

        x_val = mtd_val.convert_model_data(
            model_val,
            50,
            m_config["optimizer"],
            m_config["loss"],
            batch_size_val,
            data_dim_val,
            layer_na_fill=0,
            act_na_fill=0,
            scaler=scaler,
        ).to_numpy()
        y_val_pred_batch = batch_model.predict(x_val)
        y_val_pred_batch = y_val_pred_batch.reshape(
            y_val_pred_batch.shape[0],
        )[0]
        y_val_preds_batch.append(y_val_pred_batch * train_batch_numbers)

        y_val_pred_setup = setup_model.predict(x_val)
        y_val_pred_setup = y_val_pred_setup.reshape(
            y_val_pred_setup.shape[0],
        )[0]
        y_val_preds_setup.append(y_val_pred_setup)

    # x-axis
    x = range(len(y_val_preds_batch))

    if plot:
        # only use prediction from batch model and see error for no setup time
        plt.scatter(x, cal_score(y_val_preds_batch, real_time_process_first_batchs))
        plt.plot(x, [0.15] * len(x), c="r", linewidth=10)
        plt.plot(x, [-0.15] * len(x), c="r", linewidth=10)
        plt.title("trucated batch time error")
        plt.show()

        # see error of setup time model
        plt.scatter(
            x,
            cal_score(
                y_val_preds_setup,
                np.array(real_time_batchs) - np.array(real_time_process_first_batchs),
            ),
        )
        plt.plot(x, [0.15] * len(x), c="r", linewidth=10)
        plt.plot(x, [-0.15] * len(x), c="r", linewidth=10)
        plt.title("setup time error")
        plt.show()

        # see error for true model time prediction, combine results from batch
        # model and setup model
        plt.scatter(
            x,
            cal_score(
                np.array(y_val_preds_setup) + np.array(y_val_preds_batch),
                real_time_start_ends,
            ),
        )
        plt.plot(x, [0.15] * len(x), c="r", linewidth=10)
        plt.plot(x, [-0.15] * len(x), c="r", linewidth=10)
        plt.title("real batch time error, added pred setup time")
        plt.show()


def demo():
    # generate model configurations as data points
    data_points = 1
    df, scaler = prepare_data(data_points)

    # use data to train a ML model
    (
        x_train,
        y_batch_train,
        y_setup_train,
        x_test,
        y_batch_test,
        y_setup_test,
    ) = split_data(df)

    # build a regular dense model for batch time prediction
    batch_model = train_batch_model(
        x_train, y_batch_train, x_test, y_batch_test, plot=False
    )

    # build a dense model for setup time prediction
    setup_model = train_setup_model(
        x_train, y_setup_train, x_test, y_setup_test, plot=False
    )

    # validate tt prediction on a real case
    test_tt_prediction(batch_model, setup_model, scaler, plot=True, val_data_points=1)


if __name__ == "__main__":
    demo()
