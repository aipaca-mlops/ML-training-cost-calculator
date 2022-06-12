import copy

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.training_model.ffnn.gen_model import GenModel


class ModelLevelData:
    def __init__(
        self,
        activation_fcts: list,
        optimizers: list,
        losses: list,
        model_configs,
        input_dims=None,
        batch_sizes=None,
        epochs=None,
        truncate_from=None,
        trials=None,
        batch_strategy="random",
        input_dim_strategy="same",
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
        self.input_dims = input_dims if input_dims is not None else list(range(1, 1001))
        self.batch_sizes = (
            batch_sizes if batch_sizes is not None else [2 ** i for i in range(1, 9)]
        )
        self.epochs = epochs if epochs is not None else 10
        self.truncate_from = truncate_from if truncate_from is not None else 2
        self.trials = trials if trials is not None else 5
        self.batch_strategy = batch_strategy
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

    def convert_config_data(
        self,
        model_data,
        layer_num_upper,
        layer_na_fill=0,
        act_na_fill=0,
        opt_dummy=True,
        loss_dummy=True,
        min_max_scaler=True,
    ):
        data_rows = []
        time_rows = []

        for model_i_data in model_data:
            layer_sizes = (
                model_i_data["layer_sizes"] + [layer_na_fill] * layer_num_upper
            )
            layer_sizes = layer_sizes[:layer_num_upper]
            activations = [self.act_mapping[i] for i in model_i_data["activations"]] + [
                act_na_fill
            ] * layer_num_upper
            activations = activations[:layer_num_upper]
            if opt_dummy:
                optimizer = model_i_data["optimizer"]
            else:
                optimizer = self.opt_mapping[model_i_data["optimizer"]]
            if loss_dummy:
                loss = model_i_data["loss"]
            else:
                loss = self.loss_mapping[model_i_data["loss"]]
            batch_names = [k for k in model_i_data.keys() if k.startswith("batch_size")]

            for batch_name in batch_names:
                batch_value = int(batch_name.split("_")[-1])
                batch_time = model_i_data[batch_name]["batch_time"]
                epoch_time = model_i_data[batch_name]["epoch_time"]
                setup_time = model_i_data[batch_name]["setup_time"]
                input_dim = model_i_data[batch_name]["input_dim"]
                data_rows.append(
                    layer_sizes
                    + activations
                    + [optimizer, loss, batch_value, input_dim]
                )
                time_rows.append([batch_time, epoch_time, setup_time])

        layer_names = [f"layer_{i + 1}_size" for i in range(layer_num_upper)]
        act_names = [f"layer_{i + 1}_activation" for i in range(layer_num_upper)]
        temp_df = pd.DataFrame(
            data_rows,
            columns=layer_names
            + act_names
            + ["optimizer", "loss", "batch_size", "input_dim"],
        )
        if opt_dummy:
            first_row = dict(temp_df.iloc[0])
            for opt in self.optimizers:
                first_row["optimizer"] = opt
                first_row_df = pd.DataFrame(
                    [list(first_row.values())], columns=list(first_row.keys())
                )
                temp_df = pd.concat([temp_df, first_row_df], ignore_index=True)
            temp_df = pd.get_dummies(temp_df, columns=["optimizer"])
            temp_df = temp_df.drop(temp_df.index.tolist()[-len(self.optimizers) :])
        if loss_dummy:
            first_row = dict(temp_df.iloc[0])
            for los in self.losses:
                first_row["loss"] = los
                first_row_df = pd.DataFrame(
                    [list(first_row.values())], columns=list(first_row.keys())
                )
                temp_df = pd.concat([temp_df, first_row_df], ignore_index=True)
            temp_df = pd.get_dummies(temp_df, columns=["loss"])
            temp_df = temp_df.drop(temp_df.index.tolist()[-len(self.losses) :])
        time_df = pd.DataFrame(
            time_rows, columns=["batch_time", "epoch_time", "setup_time"]
        )
        if min_max_scaler:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(temp_df.to_numpy())
            temp_df = pd.DataFrame(scaled_data, columns=temp_df.columns)
            return pd.concat([temp_df, time_df], axis=1), scaler
        else:
            return pd.concat([temp_df, time_df], axis=1), None

    def convert_model_data(
        self,
        keras_model,
        layer_num_upper,
        optimizer,
        loss,
        batch_size,
        input_dim=None,
        layer_na_fill=0,
        act_na_fill=0,
        scaler=None,
        opt_dummy=True,
        loss_dummy=True,
    ):
        layer_sizes, acts = GenModel.get_dense_model_features(keras_model)
        if input_dim is None:
            input_dim = layer_sizes[0]
        layer_sizes = layer_sizes + [layer_na_fill] * layer_num_upper
        layer_sizes = layer_sizes[:layer_num_upper]
        acts = [self.act_mapping[i] for i in acts]
        acts = acts + [act_na_fill] * layer_num_upper
        acts = acts[:layer_num_upper]
        if opt_dummy:
            optimizer = optimizer.lower()
        else:
            optimizer = self.opt_mapping[optimizer.lower()]
        if loss_dummy:
            loss = loss.lower()
        else:
            loss = self.loss_mapping[loss.lower()]
        data = layer_sizes + acts + [optimizer, loss, batch_size, input_dim]
        layer_names = [f"layer_{i + 1}_size" for i in range(layer_num_upper)]
        act_names = [f"layer_{i + 1}_activation" for i in range(layer_num_upper)]
        temp_df = pd.DataFrame(
            [data],
            columns=layer_names
            + act_names
            + ["optimizer", "loss", "batch_size", "input_dim"],
        )
        if opt_dummy:
            first_row = dict(temp_df.iloc[0])
            for opt in self.optimizers:
                first_row["optimizer"] = opt
                first_row_df = pd.DataFrame(
                    [list(first_row.values())], columns=list(first_row.keys())
                )
                temp_df = pd.concat([temp_df, first_row_df], ignore_index=True)
            temp_df = pd.get_dummies(temp_df, columns=["optimizer"])
            temp_df = temp_df.drop(temp_df.index.tolist()[-len(self.optimizers) :])
        if loss_dummy:
            first_row = dict(temp_df.iloc[0])
            for los in self.losses:
                first_row["loss"] = los
                first_row_df = pd.DataFrame(
                    [list(first_row.values())], columns=list(first_row.keys())
                )
                temp_df = pd.concat([temp_df, first_row_df], ignore_index=True)
            temp_df = pd.get_dummies(temp_df, columns=["loss"])
            temp_df = temp_df.drop(temp_df.index.tolist()[-len(self.losses) :])

        if scaler is None:
            return temp_df
        else:
            scaled_data = scaler.transform(temp_df.to_numpy())
            return pd.DataFrame(scaled_data, columns=temp_df.columns)
