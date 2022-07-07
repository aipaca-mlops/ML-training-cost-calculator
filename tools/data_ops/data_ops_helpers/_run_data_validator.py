import json
import os
import pandas as pd
from tools.constant import EXPT_CSV_COLUMNS

from tools.util.stdout import print_err, print_ok
from tools.util.system import exit_with_err


class RunDataValidator:
    def __init__(self, run_time):
        self.run_time = run_time
        self.hardware_data_path = os.path.join("data", run_time, f"h_{run_time}.csv")
        self.expt_data_path = os.path.join("data", run_time, f"e_{run_time}.csv")
        self.model_data_folder_path = os.path.join("data", run_time, f"m_{run_time}")

    def _validate_model(self):
        if not os.path.exists(self.model_data_folder_path):
            exit_with_err(f"Model data {self.model_data_folder_path} does not exist")
        if not len(os.listdir(self.model_data_folder_path)):
            exit_with_err(f"Model data {self.model_data_folder_path} is empty")
        for f in os.listdir(self.model_data_folder_path):
            f = os.path.join(self.model_data_folder_path, f)
            if f[-5:] != ".json":
                exit_with_err(f"Model configuration file {f} is not json")
            try:
                with open(f, 'r') as j:
                    json.load(j)
            except ValueError as err:
                exit_with_err(f"Model configuration file {f} is not a valid json\nDETAILS:{err}")

    def _validate_expt(self):
        try:
            expt_df = pd.read_csv(self.expt_data_path)
        except ValueError as err:
            exit_with_err(f"Experiment table has invalid format\n{err}")
        if sorted(expt_df.columns) != sorted(EXPT_CSV_COLUMNS):
            exit_with_err(f"Experiment table should have columns {EXPT_CSV_COLUMNS}")
        if expt_df.isnull().sum().sum() > 0:
            exit_with_err(f"Experiment table contains null value")
        

    def _validate_hardware(self):
        try:
            h_df = pd.read_csv(self.hardware_data_path)
        except ValueError as err:
            exit_with_err(f"Hardware table has invalid format\nDETAILS:{err}")

        if h_df.isnull().sum().sum() > 0:
            exit_with_err(f"Hardware table contains null value")

    def validate_run_data(self):
        # TODO: the data can be checked in more details
        # in each run folder, there should be 
        # 1. a model configuration folder
        #    - all files in the folder should be json
        #    - the json files can be load without error
        # 2. an expriment csv file
        #    - the expriment table should have fixed column names == EXPT_CSV_COLUMNS
        #    - the experiment table can be read by pandas
        #    - all values should be not nan
        # 3. an hardware csv file
        #    - the experiment table can be read by pandas
        #    - all values should be not nan
        self._validate_model() 
        self._validate_expt()
        self._validate_hardware()
        path = os.path.join("data", self.run_time)
        print_ok(f"{path} is valid")