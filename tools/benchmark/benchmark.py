import base64
import hashlib
import json
import os
import re
from collections import defaultdict
from csv import DictWriter
from typing import List

import pandas as pd

from tools.constant import FILTER
from tools.constant import MODEL
from tools.util.clock import now_time_str
from tools.util.stdout import print_ok
from tools.util.stdout import print_warning


class Benchmark:
    def __init__(self, filter=FILTER):
        self.filter = self._process_filter(filter)
        self.clock = now_time_str()
        self.hardware_data_path = f"data/h_{self.clock}.csv"
        self.expt_data_path = f"data/e_{self.clock}.csv"
        self.model_data_path = f"data/m_{self.clock}"
        self.expt_csv_columns = [
            "experiment_id",
            "model_id",
            "hardware_id",
            "batch_size",
            "input_dim",
            "setup_time_ms",
            "batch_time_ms",
            "epoch_time_ms",
        ]
        self.grab_hardware_features()

    def _process_filter(self, filter):
        self.filter = defaultdict(dict)
        for k in filter:
            for item in filter[k]:
                self.filter[k][item["key"]] = item["display"]
        return self.filter

    def _filter_feature(self, feature_dict, key):
        filtered_feature_dict = {}
        # print(self.filter[key])
        for k in self.filter[key]:
            if k in feature_dict:
                filtered_feature_dict[self.filter[key][k]] = feature_dict[k]
            else:
                print_warning(f"filter {k} is not found")
        return filtered_feature_dict

    def _merge_dicts(self, *dicts):
        merged = {}
        for d in dicts:
            merged.update(d)
        return merged

    def _grab_general_hardware_features(self, key="general"):
        from tools.hardware.general_feature import GeneralFeatures

        gfeature = GeneralFeatures()
        features = self._filter_feature(gfeature.get_features(), key)
        return features

    def _grab_gpu_features(self, key="gpu"):
        from tools.hardware.gpu_feature import GpuFeatures

        gpu_features = GpuFeatures()
        features = self._filter_feature(gpu_features.get_features(), key)
        return features

    def _gen_id(self, config: dict, key):
        id = hashlib.md5(json.dumps(config).encode("utf-8")).digest()
        id_str = f"{key}_" + str(
            re.sub(r"[^a-zA-Z]", "", base64.b64encode(id).decode("utf-8"))[:5]
        )
        return id_str

    def _write_expt_csv(self, new_rows: List[dict]):
        open_new = not os.path.exists(self.expt_data_path)
        with open(self.expt_data_path, "a", newline="") as f_object:
            # Pass the CSV  file object to the Dictwriter() function
            # Result - a DictWriter object
            dictwriter_object = DictWriter(f_object, fieldnames=self.expt_csv_columns)
            # Pass the data in the dictionary as an argument into the writerow() function
            if open_new:
                dictwriter_object.writeheader()
            for new_row in new_rows:
                dictwriter_object.writerow(new_row)
            # Close the file object
            f_object.close()

    def _write_model_config_json(self, model_configs: dict):
        if not os.path.exists(self.model_data_path):
            os.mkdir(self.model_data_path)
        for m_n in model_configs:
            m_path = os.path.join(self.model_data_path, m_n + ".json")
            if os.path.exists(m_path):
                print_warning(
                    f"Overwriting model {m_n}. This might be caused by an internal error."
                )
            out_file = open(m_path, "w")
            json.dump(model_configs[m_n], out_file, indent=3)
            out_file.close()
            print_ok(f"Saved model {m_n}")

    def grab_hardware_features(self, dump=True, key="h"):
        hardware_feature = self._merge_dicts(
            self._grab_general_hardware_features(), self._grab_gpu_features()
        )
        self.hardware_id = self._gen_id(hardware_feature, key)
        hardware_features = [
            self._merge_dicts({"hardware_id": self.hardware_id}, hardware_feature)
        ]
        if dump:
            hardware_df = pd.DataFrame(
                hardware_features, index=range(len(hardware_features))
            )
            if not os.path.exists("data"):
                os.makedirs("data")
            hardware_df.to_csv(self.hardware_data_path)
            print_ok(f"Saved hardware info to {self.hardware_data_path}")

    def grab_ffnn_features(self, key="ffnn"):
        from tools.training_model.ffnn.gen_model import GenModel
        from tools.training_model.ffnn.gen_data import GenData

        gnn = GenModel()
        model_configs = gnn.generate_model_configs(
            num_model_data=MODEL[key]["num_data"]
        )
        gdata = GenData(
            model_configs,
        )
        model_data = gdata.get_train_data()
        model_ids = [self._gen_id(config, key) for config in model_configs]
        expt_data = []
        for i, m in enumerate(model_data):
            batch_names = [k for k in m.keys() if k.startswith("batch_size")]
            for b_n in batch_names:
                # assume all b_n keep in style `batch_size_xx`
                m[b_n]["batch_size"] = b_n.split("_")[-1]
                m[b_n]["model_id"] = model_ids[i]
                m[b_n]["hardware_id"] = self.hardware_id
                m[b_n]["experiment_id"] = self._gen_id(m[b_n], "e")
                expt_data.append(m[b_n])
        model_configs = {model_ids[i]: config for i, config in enumerate(model_configs)}
        self._write_expt_csv(expt_data)
        self._write_model_config_json(model_configs)
        return expt_data, model_configs

    def grab_cnn2d_features(self, key="cnn2d"):
        from tools.training_model.cnn2d.gen_data import Cnn2dModelTrainData
        from tools.training_model.cnn2d.gen_model import GenCnn2d

        gcnn2d = GenCnn2d()
        model_configs = gcnn2d.generate_model_configs(
            num_model_data=MODEL[key]["num_data"]
        )
        gdata = Cnn2dModelTrainData(model_configs)

        model_data = gdata.get_train_data()

        model_ids = [self._gen_id(config, key) for config in model_configs]
        expt_data = [m[3] for i, m in enumerate(model_data)]
        for i in range(len(expt_data)):
            expt_data[i]["model_id"] = model_ids[i]
            expt_data[i]["hardware_id"] = self.hardware_id
            expt_data[i]["experiment_id"] = self._gen_id(expt_data[i], "e")
        model_configs = {model_ids[i]: config for i, config in enumerate(model_configs)}
        self._write_expt_csv(expt_data)
        self._write_model_config_json(model_configs)
        return expt_data, model_configs

    def grab_cnn_pretrained_features(self, key="cnn_pretrained"):
        from tools.training_model.cnn_pretrained.gen_data import GenPreTrainedCnn

        expt_data = []
        model_configs = {}
        for m_n in MODEL[key]:
            gdata = GenPreTrainedCnn()
            model_data, new_model_configs = gdata.get_train_data(
                MODEL[key][m_n]["num_data"], m_n
            )

            model_ids = [self._gen_id(config, m_n) for config in new_model_configs]
            new_expt_data = model_data
            for i in range(len(new_expt_data)):
                new_expt_data[i]["model_id"] = model_ids[i]
                new_expt_data[i]["hardware_id"] = self.hardware_id
                new_expt_data[i]["experiment_id"] = self._gen_id(new_expt_data[i], "e")
            new_model_configs = {
                model_ids[i]: config for i, config in enumerate(new_model_configs)
            }
            self._write_expt_csv(new_expt_data)
            self._write_model_config_json(new_model_configs)
            expt_data += new_expt_data
            model_configs.update(new_model_configs)
        return expt_data, model_configs
