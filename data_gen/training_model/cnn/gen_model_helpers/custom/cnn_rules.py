import numpy as np


class CnnRules:
    def __init__(
        self,
        conv_layer_num_lower=1,
        conv_layer_num_upper=11,
        max_pooling_prob=0.5,
        dense_layer_num_lower=1,
        dense_layer_num_upper=6,
    ):
        # Rule: No Convolutional Layer After the First Dense Layer
        self.conv_layer_num_lower = conv_layer_num_lower
        self.conv_layer_num_upper = conv_layer_num_upper
        self.max_pooling_prob = max_pooling_prob
        self.dense_layer_num_lower = dense_layer_num_lower
        self.dense_layer_num_upper = dense_layer_num_upper

    def gen_cnn_rule(self):
        conv_layer_num = np.random.randint(
            self.conv_layer_num_lower, self.conv_layer_num_upper
        )
        dense_layer_num = np.random.randint(
            self.dense_layer_num_lower, self.dense_layer_num_upper
        )

        rule_list = []
        for _ in range(conv_layer_num):
            rule_list.append("Conv2D")
            max_pooling_appear = np.random.choice(
                [True, False],
                size=1,
                replace=True,
                p=[self.max_pooling_prob, 1 - self.max_pooling_prob],
            )[0]
            if max_pooling_appear:
                rule_list.append("MaxPooling2D")

        rule_list.append("Flatten")

        rule_list.extend(["Dense"] * dense_layer_num)

        return rule_list
