from tools.training_model.cnn2d.gen_data import GenData
from tools.training_model.cnn2d.gen_model import GenModel


def demo(data_points=10):
    gen = GenModel()
    model_configs = gen.generate_model_configs(num_model_data=data_points)
    mtd = GenData(model_configs)

    model_data = mtd.get_train_data()
    print(model_data[0])


if __name__ == "__main__":
    demo()
