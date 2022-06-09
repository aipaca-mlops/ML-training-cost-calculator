import subprocess

import tensorflow as tf
import yaml


with open("data_gen/training_model/ffnn/config.yaml", "r") as stream:
    try:
        FFNN_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open("data_gen/training_model/cnn/cnn2d/config.yaml", "r") as stream:
    try:
        CNN2D_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open("data_gen/training_model/cnn/pretrained/config.yaml", "r") as stream:
    try:
        CNN_PRETRAINED_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

