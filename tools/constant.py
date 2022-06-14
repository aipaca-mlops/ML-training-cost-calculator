import yaml


with open("tools/training_model/ffnn/config.yaml", "r") as stream:
    try:
        FFNN_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open("tools/training_model/cnn2d/config.yaml", "r") as stream:
    try:
        CNN2D_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open("tools/training_model/cnn_pretrained/config.yaml", "r") as stream:
    try:
        CNN_PRETRAINED_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open("tools/training_model/rnn/config.yaml", "r") as stream:
    try:
        RNN_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open("tools/benchmark/benchmark_settings.yaml", "r") as stream:
    try:
        benchmark_settings = yaml.safe_load(stream)
        FILTER = benchmark_settings["filter"]
        MODEL = benchmark_settings["model"]
    except yaml.YAMLError as exc:
        print(exc)


TIME_COLUMNS = ["batch_time_ms", "epoch_time_ms", "setup_time_ms"]
