import yaml


with open("src/training_model/ffnn/config.yaml", "r") as stream:
    try:
        FFNN_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open("src/training_model/cnn2d/config.yaml", "r") as stream:
    try:
        CNN2D_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open("src/training_model/cnn_pretrained/config.yaml", "r") as stream:
    try:
        CNN_PRETRAINED_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open("src/training_model/rnn/config.yaml", "r") as stream:
    try:
        RNN_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
