import base64
import hashlib
import json
import os
import re
from collections import defaultdict
from csv import DictWriter
from typing import List

import pandas as pd

from tools.util.stdout import print_ok
from tools.util.stdout import print_warning


class BenchmarkTools:
    def __init__(
        self,
        filter,
        hardware_data_path,
        expt_data_path,
        model_data_path,
        expt_csv_columns,
    ):
        self.filter = self._process_filter(filter)
        self.hardware_data_path = hardware_data_path
        self.expt_data_path = expt_data_path
        self.model_data_path = model_data_path
        self.expt_csv_columns = expt_csv_columns

    def _process_filter(self, filter):
        self.filter = defaultdict(dict)
        for k in filter:
            for item in filter[k]:
                self.filter[k][item["key"]] = item["display"]
        return self.filter

    def _filter_feature(self, feature_dict, key):
        filtered_feature_dict = {}
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

    def _gen_id(self, config: dict, key):
        id = hashlib.md5(json.dumps(config).encode("utf-8")).digest()
        id_str = f"{key}_" + str(
            re.sub(r"[^a-zA-Z]", "", base64.b64encode(id).decode("utf-8"))[:5]
        )
        return id_str

    def _write_hardware_csv(self, hardware_features):
        hardware_df = pd.DataFrame(
            hardware_features, index=range(len(hardware_features))
        )
        if not os.path.exists("data"):
            os.makedirs("data")
        hardware_df.to_csv(self.hardware_data_path)
        print_ok(f"Saved hardware info to {self.hardware_data_path}")

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
