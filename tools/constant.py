import os

import yaml
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "production")

print(f"Running {ENV} environment")

with open(f"tools/training_model/ffnn/config/config_{ENV}.yaml", "r") as stream:
    try:
        FFNN_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open(f"tools/training_model/cnn2d/config/config_{ENV}.yaml", "r") as stream:
    try:
        CNN2D_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open(
    f"tools/training_model/cnn_pretrained/config/config_{ENV}.yaml", "r"
) as stream:
    try:
        CNN_PRETRAINED_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open(f"tools/training_model/rnn/config/config_{ENV}.yaml", "r") as stream:
    try:
        RNN_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open(
    f"tools/benchmark/benchmark_settings/benchmark_settings_{ENV}.yaml", "r"
) as stream:
    try:
        benchmark_settings = yaml.safe_load(stream)
        FILTER = benchmark_settings["filter"]
        MODEL = benchmark_settings["model"]
    except yaml.YAMLError as exc:
        print(exc)


TIME_COLUMNS = ["batch_time_ms", "epoch_time_ms", "setup_time_ms"]

EXPT_CSV_COLUMNS = [
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

S3_REGION = "us-east-1"