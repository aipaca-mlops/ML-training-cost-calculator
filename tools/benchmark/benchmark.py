import os

from tools.benchmark.benchmark_helpers._benchmark_tools import BenchmarkTools
from tools.constant import FILTER
from tools.constant import MODEL
from tools.util.clock import now_time_str
from tools.util.stdout import print_warning


class Benchmark(BenchmarkTools):
    # import helper methods
    from tools.benchmark.benchmark_helpers._benchmark_hardware import (
        grab_hardware_features,
        _grab_general_hardware_features,
        _grab_gpu_features,
    )
    from tools.benchmark.benchmark_helpers._benchmark_model import (
        grab_ffnn_features,
        grab_cnn2d_features,
        grab_cnn_pretrained_features,
        grab_rnn_features,
    )

    def __init__(self, filter=FILTER):
        self.clock = now_time_str()
        hardware_data_path = os.path.join("data", self.clock, f"h_{self.clock}.csv")
        expt_data_path = os.path.join("data", self.clock, f"e_{self.clock}.csv")
        model_data_path = os.path.join("data", self.clock, f"m_{self.clock}.csv")
        expt_csv_columns = [
            "experiment_id",
            "model_id",
            "hardware_id",
            "model_type",
            "batch_size",
            "input_dim",
            "setup_time_ms",
            "batch_time_ms",
            "epoch_time_ms",
        ]
        filter = filter
        self.grab_which_model = {
            "ffnn": self.grab_ffnn_features,
            "cnn2d": self.grab_cnn2d_features,
            "cnn_pretrained": self.grab_cnn_pretrained_features,
            "rnn": self.grab_rnn_features,
        }
        BenchmarkTools.__init__(
            self,
            filter,
            hardware_data_path,
            expt_data_path,
            model_data_path,
            expt_csv_columns,
        )

    def run(self, model: dict = MODEL):
        self._update_clock()
        self._create_data_folder()
        self.grab_hardware_features()
        for m_n in model:
            if m_n not in self.grab_which_model:
                print_warning(
                    f"Model {m_n} is not found. Please choose one from {list(self.grab_which_model.keys())}"
                )
                continue
            self.grab_which_model[m_n]()
