from collections import defaultdict

from tools.constant import MODEL


def grab_ffnn_features(self, key="ffnn"):
    from tools.training_model.ffnn.gen_model import GenModel
    from tools.training_model.ffnn.gen_data import GenData

    gnn = GenModel()
    model_configs = gnn.generate_model_configs(num_model_data=MODEL[key]["num_data"])
    gdata = GenData(
        model_configs,
    )
    model_data = gdata.get_train_data()
    model_ids = [self._gen_id(config, key) for config in model_configs]
    expt_data = defaultdict(dict)
    ctr = 0
    for i, m in enumerate(model_data):
        batch_names = [k for k in m.keys() if k.startswith("batch_size")]
        for b_n in batch_names:
            for n in self.expt_csv_columns:
                if n in model_data[i][b_n]:
                    expt_data[i][n] = model_data[i][b_n][n]
            # assume all b_n keep in style `batch_size_xx`
            expt_data[ctr]["batch_size"] = b_n.split("_")[-1]
            expt_data[ctr]["model_id"] = model_ids[i]
            expt_data[ctr]["hardware_id"] = self.hardware_id
            expt_data[ctr]["experiment_id"] = self._gen_id(m[b_n], "e")
            expt_data[ctr]["model_type"] = key
            ctr += 1
    expt_data = list(expt_data.values())
    model_configs = {model_ids[i]: config for i, config in enumerate(model_configs)}
    self._write_expt_csv(expt_data)
    self._write_model_config_json(model_configs)
    return expt_data, model_configs


def grab_cnn2d_features(self, key="cnn2d"):
    from tools.training_model.cnn2d.gen_data import GenData
    from tools.training_model.cnn2d.gen_model import GenModel

    gcnn2d = GenModel()
    model_configs = gcnn2d.generate_model_configs(num_model_data=MODEL[key]["num_data"])
    gdata = GenData(model_configs)

    model_data = gdata.get_train_data()

    model_ids = [self._gen_id(config, key) for config in model_configs]
    expt_data = defaultdict(dict)
    for i in range(len(model_data)):
        for n in self.expt_csv_columns:
            if n in model_data[i][3]:
                expt_data[i][n] = model_data[i][3][n]
        expt_data[i]["model_id"] = model_ids[i]
        expt_data[i]["hardware_id"] = self.hardware_id
        expt_data[i]["model_type"] = key
        expt_data[i]["experiment_id"] = self._gen_id(expt_data[i], "e")
    expt_data = list(expt_data.values())
    model_configs = {model_ids[i]: config for i, config in enumerate(model_configs)}
    self._write_expt_csv(expt_data)
    self._write_model_config_json(model_configs)
    return expt_data, model_configs


def grab_cnn_pretrained_features(self, key="cnn_pretrained"):
    from tools.training_model.cnn_pretrained.gen_data import GenData

    expt_data = []
    model_configs = {}
    for m_n in MODEL[key]:
        gdata = GenData()
        model_data, new_model_configs = gdata.get_train_data(
            MODEL[key][m_n]["num_data"], m_n
        )

        model_ids = [self._gen_id(config, m_n) for config in new_model_configs]
        new_expt_data = defaultdict(dict)
        for i in range(len(model_data)):
            for n in self.expt_csv_columns:
                if n in model_data[i]:
                    new_expt_data[i][n] = model_data[i][n]
            new_expt_data[i]["model_id"] = model_ids[i]
            new_expt_data[i]["hardware_id"] = self.hardware_id
            new_expt_data[i]["model_type"] = m_n
            new_expt_data[i]["experiment_id"] = self._gen_id(new_expt_data[i], "e")
        new_model_configs = {
            model_ids[i]: config for i, config in enumerate(new_model_configs)
        }
        new_expt_data = list(new_expt_data.values())
        self._write_expt_csv(new_expt_data)
        self._write_model_config_json(new_model_configs)
        expt_data += new_expt_data
        model_configs.update(new_model_configs)
    return expt_data, model_configs


def grab_rnn_features(self, key="rnn"):
    from tools.training_model.rnn.gen_data import GenData
    from tools.training_model.rnn.gen_model import GenModel

    g = GenModel()
    model_configs = g.generate_model_configs(MODEL[key]["num_data"])
    mtd = GenData(model_configs)
    model_data = mtd.get_train_data()

    model_ids = [self._gen_id(config, key) for config in model_configs]
    expt_data = defaultdict(dict)
    for i in range(len(model_data)):
        for n in self.expt_csv_columns:
            if n in model_data[i]:
                expt_data[i][n] = model_data[i][n]
        expt_data[i]["model_id"] = model_ids[i]
        expt_data[i]["hardware_id"] = self.hardware_id
        expt_data[i]["model_type"] = key
        expt_data[i]["experiment_id"] = self._gen_id(expt_data[i], "e")
    expt_data = list(expt_data.values())
    model_configs = {model_ids[i]: config for i, config in enumerate(model_configs)}
    self._write_expt_csv(expt_data)
    self._write_model_config_json(model_configs)
    return expt_data, model_configs
