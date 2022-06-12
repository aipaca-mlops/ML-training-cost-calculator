from src.training_model.cnn_pretrained.gen_data import (
    GenPreTrainedCnn,
)
from src.training_model.constant import CNN_PRETRAINED_CONFIG


def demo_pretrained_models():
    """
    check here for all valid models: https://www.tensorflow.org/api_docs/python/tf/keras/applications
    :return:
    """
    cmtd = GenPreTrainedCnn()
    num_data = CNN_PRETRAINED_CONFIG["num_data"]
    model_name = CNN_PRETRAINED_CONFIG["model_name"]
    model_data = cmtd.get_train_data(num_data, model_name)
    print(model_data[0])


if __name__ == "__main__":
    demo_pretrained_models()
