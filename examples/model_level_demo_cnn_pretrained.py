from tools.training_model.cnn_pretrained.gen_data import (
    GenPreTrainedCnn,
)


def demo_pretrained_models():
    """
    check here for all valid models: https://www.tensorflow.org/api_docs/python/tf/keras/applications
    :return:
    """
    cmtd = GenPreTrainedCnn()
    model_data, model_configs = cmtd.get_train_data(1, "VGG16")
    print(model_data[0])


if __name__ == "__main__":
    demo_pretrained_models()
