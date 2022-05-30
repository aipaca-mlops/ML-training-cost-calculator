"""
****************************************
 * @author: Xin Zhang
 * Date: 8/26/21
****************************************
"""
"""
****************************************
 * @author: Xin Zhang
 * Date: 5/22/21
****************************************
"""
import time
import tensorflow.keras as keras
import pandas as pd
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional
from random import sample
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import copy

activation_fcts = [
    'relu', "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"
]
optimizers = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]
losses = ["mae", "mape", "mse", "msle", "poisson", "categorical_crossentropy"]
rnn_layer_types = ['SimpleRNN', 'LSTM', 'GRU']


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


class gen_bidirectional_rnn:
    def __init__(
        self,
        rnn_layers_num_lower=1,
        rnn_layers_num_upper=10,
        rnn_layer_size_lower=1,
        rnn_layer_size_upper=101,
        dense_layers_num_lower=1,
        dense_layers_num_upper=3,
        dense_layer_size_lower=1,
        dense_layer_size_upper=6,
        activation='random',
        optimizer='random',
        loss='random',
        rnn_layer_type='random'
    ):
        self.rnn_layers_num_lower = rnn_layers_num_lower
        self.rnn_layers_num_upper = rnn_layers_num_upper
        self.rnn_layer_size_lower = rnn_layer_size_lower
        self.rnn_layer_size_upper = rnn_layer_size_upper
        self.dense_layers_num_lower = dense_layers_num_lower,
        self.dense_layers_num_upper = dense_layers_num_upper,
        self.dense_layer_size_lower = dense_layer_size_lower,
        self.dense_layer_size_upper = dense_layer_size_upper,
        self.activation_pick = activation
        self.optimizer_pick = optimizer
        self.loss_pick = loss
        self.rnn_layer_type_pick = rnn_layer_type
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
        if layer_type == 'SimpleRNN':
            rnn_layer = SimpleRNN
        if layer_type == 'LSTM':
            rnn_layer = LSTM
        if layer_type == 'GRU':
            rnn_layer = GRU

        model = Sequential()
        for index, size in enumerate(rnn_layer_sizes + dense_layer_sizes):
            if index < len(rnn_layer_sizes) - 1:
                model.add(
                    Bidirectional(
                        rnn_layer(units=size, activation=activations[index], return_sequences=True)
                    )
                )
            elif index == len(rnn_layer_sizes) - 1:
                model.add(Bidirectional(rnn_layer(units=size, activation=activations[index])))
            else:
                model.add(Dense(units=size, activation=activations[index]))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def get_RNN_model_features(keras_model):
        layers = [
            layer_info for layer_info in keras_model.get_config()['layers']
            if layer_info['class_name'] == 'Bidirectional' or layer_info['class_name'] == 'Dense'
        ]
        layer_sizes = []
        acts = []
        layer_Type = []
        for l in layers:
            if l['class_name'] == 'Dense':
                layer_sizes.append(l['config']['units'])
                acts.append(l['config']['activation'])
                layer_Type.append(l['class_name'])
            else:
                layer_sizes.append(l['config']['layer']['config']['units'])
                acts.append(l['config']['layer']['config']['activation'])
                layer_Type.append(l['config']['layer']['class_name'])
        return layer_sizes, acts, layer_Type

    def generate_model(self):
        rnn_layers_num = np.random.randint(self.rnn_layers_num_lower, self.rnn_layers_num_upper)
        rnn_layer_sizes = np.random.randint(
            self.rnn_layer_size_lower, self.rnn_layer_size_upper, rnn_layers_num
        )
        dense_layers_num = np.random.randint(
            self.dense_layers_num_lower, self.dense_layers_num_upper
        )
        dense_layer_sizes = np.random.randint(
            self.dense_layer_size_lower, self.dense_layer_size_upper, dense_layers_num
        )

        if self.activation_pick == 'random':
            activations = np.random.choice(self.activation_fcts, rnn_layers_num + dense_layers_num)
        else:
            activations = np.random.choice([self.activation_pick],
                                           rnn_layers_num + dense_layers_num)
        if self.optimizer_pick == 'random':
            optimizer = np.random.choice(self.optimizers)
        else:
            optimizer = self.optimizer_pick
        if self.loss_pick == 'random':
            loss = np.random.choice(self.losses)
        else:
            loss = self.loss_pick
        if self.rnn_layer_type_pick == 'random':
            rnn_layer = np.random.choice(self.rnn_layer_types)
        else:
            rnn_layer = self.rnn_layer_type_pick

        return {
            #'model': gen_bidirectional_rnn.build_RNN_model(rnn_layer, list(rnn_layer_sizes), list(dense_layer_sizes), activations, optimizer, loss),
            'rnn_layer_sizes': [int(i) for i in rnn_layer_sizes],
            'dense_layer_sizes': [int(i) for i in dense_layer_sizes],
            'activations': list(activations),
            'optimizer': optimizer,
            'loss': loss,
            'rnn_type': rnn_layer,
        }

    def generate_model_configs(self, num_model_data=1000, progress=True):
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = gen_bidirectional_rnn.nothing
        for i in loop_fun(range(num_model_data)):
            data = self.generate_model()
            #del data['model']
            model_configs.append(data)
        return model_configs


class model_train_data:
    def __init__(
        self,
        model_configs,
        input_dims=None,
        batch_sizes=None,
        epochs=None,
        truncate_from=None,
        trials=None,
        input_dim_strategy='same'
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
        self.model_configs = []
        for info_dict in model_configs:
            d2 = copy.deepcopy(info_dict)
            self.model_configs.append(d2)
        self.input_dims = input_dims if input_dims is not None else list(range(1, 101))
        self.batch_sizes = batch_sizes if batch_sizes is not None else [2**i for i in range(1, 9)]
        self.epochs = epochs if epochs is not None else 10
        self.truncate_from = truncate_from if truncate_from is not None else 2
        self.trials = trials if trials is not None else 5
        self.input_dim_strategy = input_dim_strategy
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses
        self.act_mapping = dict((act, index + 1) for index, act in enumerate(self.activation_fcts))
        self.opt_mapping = dict((opt, index + 1) for index, opt in enumerate(self.optimizers))
        self.loss_mapping = dict((loss, index + 1) for index, loss in enumerate(self.losses))

    def get_train_data(self, progress=True):
        model_data = []
        model_configs = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = gen_bidirectional_rnn.nothing
        for info_dict in self.model_configs:
            d2 = copy.deepcopy(info_dict)
            model_configs.append(d2)
        for model_config in loop_fun(model_configs):
            model = gen_bidirectional_rnn.build_RNN_model(
                layer_type=model_config['rnn_type'],
                rnn_layer_sizes=model_config['rnn_layer_sizes'],
                dense_layer_sizes=model_config['dense_layer_sizes'],
                activations=model_config['activations'],
                optimizer=model_config['optimizer'],
                loss=model_config['loss']
            )
            batch_sizes = sample(self.batch_sizes, 1)
            input_dim = sample(self.input_dims, 1)[0]
            for batch_size in batch_sizes:
                batch_size_data_batch = []
                batch_size_data_epoch = []
                if self.input_dim_strategy == 'same':
                    try:
                        input_shape = model.get_config(
                        )['layers'][0]['config']['layer']['config']['units']
                    except:
                        input_shape = model.get_config(
                        )['layers'][0]['config']['batch_input_shape'][2]
                else:
                    input_shape = input_dim
                out_shape = model.get_config()['layers'][-1]['config']['units']
                x = np.ones((batch_size, 1, input_shape), dtype=np.float32)
                y = np.ones((batch_size, out_shape), dtype=np.float32)
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

                batch_times_truncated = batch_size_data_batch[self.truncate_from:]
                epoch_times_trancuted = batch_size_data_epoch[self.truncate_from:]
                recovered_time = [
                    np.median(batch_times_truncated)
                ] * self.truncate_from + batch_times_truncated

                model_config['batch_size'] = batch_size
                model_config['batch_time'] = np.median(batch_times_truncated)
                model_config['epoch_time'] = np.median(epoch_times_trancuted)
                model_config['setup_time'] = np.sum(batch_size_data_batch) - sum(recovered_time)
                model_config['input_dim'] = input_shape
            model_data.append(model_config)
        return model_data


class convert_bidirectional_rnn_data:
    def __init__(self):
        self.optimizers = optimizers
        self.rnn_layer_types = rnn_layer_types

        unique_all_rnns = sorted(list(set(self.rnn_layer_types)))
        unique_all_optimizers = sorted(list(set(self.optimizers)))

        opt_enc = OneHotEncoder(handle_unknown='ignore')
        rnn_enc = OneHotEncoder(handle_unknown='ignore')

        x_rnns = [[i] for i in unique_all_rnns]
        rnn_enc.fit(x_rnns)
        x_opts = [[i] for i in unique_all_optimizers]
        opt_enc.fit(x_opts)

        self.rnn_enc = rnn_enc
        self.opt_enc = opt_enc

    @staticmethod
    def get_rnn_type(model):
        return [
            i['config']['layer']['class_name'] for i in model.get_config()['layers']
            if i['class_name'] == 'Bidirectional'
        ][0]

    @staticmethod
    def get_units_sum_rnn_keras(model_obj):
        layers = [
            layer_info for layer_info in model_obj.get_config()['layers']
            if layer_info['class_name'] == 'Bidirectional'
        ]
        layer_sizes = []
        for l in layers:
            layer_sizes.append(l['config']['layer']['config']['units'])
        return sum(layer_sizes)

    @staticmethod
    def get_units_sum_dense_keras(model_obj):
        return sum([
            layer['config']['units'] for layer in model_obj.get_config()['layers']
            if layer['class_name'] == 'Dense'
        ])

    def convert_model_config(self, model_config_rnn, data_type='Units', min_max_scaler=True):
        """

        @param model_config_dense:
        @param data_type: str "Units" or "FLOPs"
        @param min_max_scaler:
        @return:
        """
        if data_type.lower().startswith('f'):
            print('currently FLOPs is not avaliable for RNN')
        all_batch_sizes = []
        all_optimizers = []
        all_rnn_types = []
        flops_data = []
        dense_units_data = []
        rnn_units_data = []
        times_data = []
        for index, model_config in enumerate(tqdm(model_config_rnn)):
            batch_size = model_config['batch_size']
            all_batch_sizes.append(batch_size)
            all_optimizers.append(model_config['optimizer'])
            all_rnn_types.append(model_config['rnn_type'])
            dense_units_data.append(sum(model_config['dense_layer_sizes']))
            rnn_units_data.append(sum(model_config['rnn_layer_sizes']))
            times_data.append(model_config['batch_time'])

        rnn_data = []
        for rnn_size, dense_size, batch, opt, rnn_type in tqdm(list(zip(rnn_units_data,
                                                                        dense_units_data,
                                                                        all_batch_sizes,
                                                                        all_optimizers,
                                                                        all_rnn_types))):
            optimizer_onehot = list(self.opt_enc.transform([[opt]]).toarray()[0])
            rnn_type_onehot = list(self.rnn_enc.transform([[rnn_type]]).toarray()[0])
            rnn_data.append([rnn_size, dense_size, batch] + optimizer_onehot + rnn_type_onehot)

        if min_max_scaler:
            scaler = MinMaxScaler()
            scaler.fit(rnn_data)
            scaler_rnn_data = scaler.transform(rnn_data)
            return scaler_rnn_data, np.array(times_data), scaler
        else:
            return rnn_data, np.array(times_data), None

    def convert_model_keras(
        self, rnn_model_obj, optimizer, batch_size, data_type='Unit', scaler=None
    ):
        rnn_type = convert_bidirectional_rnn_data.get_rnn_type(rnn_model_obj)
        dense_unit_sum = convert_bidirectional_rnn_data.get_units_sum_dense_keras(rnn_model_obj)
        rnn_unit_sum = convert_bidirectional_rnn_data.get_units_sum_rnn_keras(rnn_model_obj)

        optimizer_onehot = list(self.opt_enc.transform([[optimizer]]).toarray()[0])
        rnn_type_onehot = list(self.rnn_enc.transform([[rnn_type]]).toarray()[0])

        layer_data = [rnn_unit_sum, dense_unit_sum, batch_size] + optimizer_onehot + rnn_type_onehot

        if scaler is not None:
            scaled_data = scaler.transform(np.array([layer_data]))
            return scaled_data
        else:
            return layer_data
