from data_gen.training_model.cnn.gen_data import Cnn2dModelTrainData
from data_gen.training_model.cnn.gen_model import GenCnn2d
from data_gen.training_model.cnn.gen_model_helpers.pretrained.gen_pre_trained_cnn import (
    GenPreTrainedCnn,
)


def demo():
    data_points = 10
    gen = GenCnn2d(
        input_shape_lower=20,
        input_shape_upper=101,
    )
    model_configs = gen.generate_model_configs(
        num_model_data=data_points, progress=True
    )
    mtd = Cnn2dModelTrainData(
        model_configs, batch_sizes=None, epochs=None, truncate_from=None, trials=None
    )

    model_data = mtd.get_train_data(progress=True)
    print(model_data[0])


def demo_classic_models():
    """
    check here for all valid models: https://www.tensorflow.org/api_docs/python/tf/keras/applications
    :return:
    """
    cmtd = GenPreTrainedCnn(
        batch_sizes=[2, 4],
        optimizers=["sgd", "rmsprop", "adam"],
        losses=["mse", "msle", "poisson", "categorical_crossentropy"],
    )
    model_data = cmtd.get_train_data(
        "VGG16", input_shape=None, output_size=1000, progress=True
    )
    mtd = Cnn2dModelTrainData(
        model_data, batch_sizes=None, epochs=None, truncate_from=None, trials=None
    )
    model_data = mtd.get_train_data(progress=True)
    print(model_data[0])


if __name__ == "__main__":
    demo()
