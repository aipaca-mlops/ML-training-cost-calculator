from tools.constant import RNN_CONFIG
from tools.training_model.rnn.gen_data import GenData
from tools.training_model.rnn.gen_model import GenModel


def demo():
    # Note that RNN models not use cuDNN kernels since it doesn't meet the criteria.
    # It will use a generic GPU kernel as fallback when running on GPU.
    g = GenModel()
    model_configs = g.generate_model_configs(RNN_CONFIG["num_data"])
    mtd = GenData(model_configs)
    model_data = mtd.get_train_data()
    rnn_data, times_data, scaler = mtd.convert_model_config(model_data)
    print(rnn_data)


if __name__ == "__main__":
    demo()
