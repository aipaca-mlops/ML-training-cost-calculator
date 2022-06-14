from tools.training_model.cnn2d.gen_data import Cnn2dModelTrainData
from tools.training_model.cnn2d.gen_model import GenCnn2d


def demo(data_points=10):
    gen = GenCnn2d()
    model_configs = gen.generate_model_configs(num_model_data=data_points)
    mtd = Cnn2dModelTrainData(model_configs)

    model_data = mtd.get_train_data()
    print(model_data[0])


if __name__ == "__main__":
    demo()
