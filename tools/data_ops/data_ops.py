import os

from tools.data_ops.data_ops_helpers._run_data_validator import RunDataValidator
from tools.data_ops.data_ops_helpers._s3_service import S3Service


class DataOps(S3Service):
    def find_last_run(self):
        runs = [r for r in os.listdir("data") if os.path.isdir(os.path.join("data", r))]
        last_run_time = sorted(runs)[-1]
        return last_run_time

    def list_run_files(self, run_time):
        hardware_data_path = os.path.join("data", run_time, f"h_{run_time}.csv")
        expt_data_path = os.path.join("data", run_time, f"e_{run_time}.csv")
        model_data_folder_path = os.path.join("data", run_time, f"m_{run_time}")
        res = [hardware_data_path, expt_data_path]
        for m in os.listdir(model_data_folder_path):
            m_config_path = os.path.join(model_data_folder_path, m)
            res.append(m_config_path)
        return res

    def upload_to_s3(self, run_time=None):
        # if last_run == False, upload all data
        if not run_time:
            run_time = self.find_last_run()
        rdv = RunDataValidator(run_time)
        rdv.validate_run_data()
        # start uploading data
        for f in self.list_run_files(run_time):
            self.upload_file_to_s3(f, )

data_ops = DataOps()
data_ops.list_run_files(data_ops.find_last_run())
