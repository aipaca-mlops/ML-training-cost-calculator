import subprocess

import tensorflow as tf
import yaml

from data_gen.training_model.util.dummy_context_mgr import dummy_context_mgr


with open("data_gen/training_model/ffnn/config.yaml", "r") as stream:
    try:
        FFNN_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open("data_gen/training_model/cnn/config.yaml", "r") as stream:
    try:
        CNN_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

IS_M1 = (
    subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    .decode("utf-8")
    .startswith("Apple M1")
)

# Since M1 GPU does not support all tf operation, we used the TRAINING_CONTEXT_MGR to limit device to only cpu
TRAINING_CONTEXT_MGR = dummy_context_mgr() if not IS_M1 else tf.device("/cpu:0")
