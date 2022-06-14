import collections
import random

from tools.constant import CNN2D_CONFIG


class BuildModel:
    def __init__(
        self,
        DEFAULT_INPUT_SHAPE=(32, 32, 3),
        filter_lower=1,
        filter_upper=101,
        paddings=None,
        dense_lower=1,
        dense_upper=1001,
        activations=None,
        optimizers=None,
        losses=None,
        activation_fcts=CNN2D_CONFIG["activation_fcts"],
    ):
        self.kwargs_list: list
        self.layer_orders: list
        self.DEFAULT_INPUT_SHAPE = DEFAULT_INPUT_SHAPE
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses
        self.paddings = paddings

        OPTIONS = collections.defaultdict(dict)

        OPTIONS["Model"]["layer"] = [
            "Conv2D",
            "Dense",
            "MaxPooling2D",
            "Dropout",
            "Flatten",
        ]  # the model's layer can be either Conv2D or Dense
        OPTIONS["Compile"]["optimizer"] = (
            optimizers if optimizers is not None else self.optimizers.copy()
        )
        OPTIONS["Compile"]["loss"] = (
            losses if losses is not None else self.losses.copy()
        )
        OPTIONS["Dense"]["units"] = list(range(dense_lower, dense_upper))
        OPTIONS["Dense"]["activation"] = (
            activations if activations is not None else self.activation_fcts.copy()
        )
        OPTIONS["Conv2D"]["filters"] = list(range(filter_lower, filter_upper))
        OPTIONS["Conv2D"]["padding"] = (
            paddings if paddings is not None else self.paddings.copy()
        )
        OPTIONS["Conv2D"]["activation"] = (
            activations if activations is not None else self.activation_fcts.copy()
        )
        OPTIONS["MaxPooling2D"]["padding"] = (
            paddings if paddings is not None else self.paddings.copy()
        )
        OPTIONS["Dropout"]["rate"] = [0.1]

        self.options = OPTIONS

    def chooseRandomComb(self, options_layer, activations=None):
        res = dict()
        for k, v in options_layer.items():
            if k == "activation" and activations is not None:
                res[k] = random.choice(activations)
            else:
                res[k] = random.sample(v, 1)[0]
        return res

    def generateRandomModelConfigList(  # noqa: C901
        self, layer_orders, input_shape=None
    ):
        if input_shape is None:
            input_shape = self.DEFAULT_INPUT_SHAPE

        def updateImageShape(_l, _kwargs, _image_shape):
            kernel_size: tuple
            if _l == "Conv2D":
                if (
                    type(_kwargs["kernel_size"]) == int
                ):  # when kwargs["kernel_size"] was set by int
                    kernel_size = (_kwargs["kernel_size"], _kwargs["kernel_size"])
                else:
                    kernel_size = _kwargs["kernel_size"]
            elif _l == "MaxPooling2D":
                if (
                    type(_kwargs["pool_size"]) == int
                ):  # when kwargs["kernel_size"] was set by int
                    # for program simplicity, I called pool_size as kernel_size
                    kernel_size = (_kwargs["pool_size"], _kwargs["pool_size"])
                else:
                    kernel_size = _kwargs["pool_size"]

            if type(_kwargs["strides"]) == int:  # when kwargs["strides"] was set by int
                strides = (_kwargs["strides"], _kwargs["strides"])
            else:
                strides = _kwargs["strides"]
            if _kwargs["padding"] == "valid":
                _image_shape[0] = (_image_shape[0] - kernel_size[0]) // strides[0] + 1
                _image_shape[1] = (_image_shape[1] - kernel_size[1]) // strides[1] + 1
            if _kwargs["padding"] == "same":
                if _image_shape[0] % strides[0] > 0:
                    _image_shape[0] = _image_shape[0] // strides[0] + 1
                else:
                    _image_shape[0] = _image_shape[0] // strides[0]
                if _image_shape[1] % strides[1] > 0:
                    _image_shape[1] = _image_shape[1] // strides[1] + 1
                else:
                    _image_shape[1] = _image_shape[1] // strides[1]
            assert _image_shape[0] > 0 and _image_shape[1] > 0
            return _image_shape

        def validKernelStridesSize(_l, _kwargs, _image_shape):
            if _l == "Conv2D":
                if type(_kwargs["kernel_size"]) == int:
                    kernel_size = (_kwargs["kernel_size"], _kwargs["kernel_size"])
                else:
                    kernel_size = _kwargs["kernel_size"]
            elif _l == "MaxPooling2D":
                if (
                    type(_kwargs["pool_size"]) == int
                ):  # when kwargs["kernel_size"] was set by int
                    # for program simplicity, I called pool_size as kernel_size
                    kernel_size = (_kwargs["pool_size"], _kwargs["pool_size"])
                else:
                    kernel_size = _kwargs["pool_size"]

            if type(_kwargs["strides"]) == int:
                strides = (_kwargs["strides"], _kwargs["strides"])
            else:
                strides = _kwargs["strides"]
            judge = True
            if _l in ["Conv2D", "MaxPooling2D"]:
                judge = judge and (
                    kernel_size[0] <= _image_shape[0]
                    and kernel_size[1] <= _image_shape[1]
                )
            judge = judge and (
                strides[0] <= _image_shape[0] and strides[1] <= _image_shape[1]
            )
            if judge:
                return True
            else:
                return False

        options = self.options
        kwargs_list = []
        image_shape: list = list(input_shape[:2])
        image_shape_list: list = []
        # image_shape should end up in the same shape as model
        new_layer_orders = []
        max_strides = [3, 3]

        for i, lo in enumerate(layer_orders):
            if lo == "Dense":
                kwargs = self.chooseRandomComb(
                    options["Dense"], options["Dense"]["activation"]
                )
            elif lo == "Conv2D":
                if image_shape[0] == 1 or image_shape[1] == 1:
                    # if one of the image dim has only size one, we stop adding new conv2D
                    continue
                options_conv2d = options["Conv2D"].copy()
                # always ensure the kernel and strides size is smaller than the image
                options_conv2d["kernel_size"] = list(
                    zip(range(1, image_shape[0]), range(1, image_shape[1]))
                )

                options_conv2d["strides"] = [(1, 1)] * 10 + list(
                    zip(range(1, max_strides[0]), range(1, max_strides[1]))
                )
                kwargs = self.chooseRandomComb(options_conv2d)
                image_shape = updateImageShape(lo, kwargs, image_shape)
                max_strides = [
                    min(max_strides[0], max(1, image_shape[0])),
                    min(max_strides[1], max(1, image_shape[1])),
                ]
            elif lo == "MaxPooling2D":
                if image_shape[0] == 1 or image_shape[1] == 1:
                    # if one of the image dim has only size one, we stop adding new conv2D
                    continue
                options_maxpooling2d = options["MaxPooling2D"].copy()
                options_maxpooling2d["pool_size"] = list(
                    zip(range(1, image_shape[0]), range(1, image_shape[1]))
                )
                options_maxpooling2d["strides"] = [(1, 1)] * 10 + list(
                    zip(range(1, max_strides[0]), range(1, max_strides[1]))
                )
                kwargs = self.chooseRandomComb(options_maxpooling2d)
                image_shape = updateImageShape(lo, kwargs, image_shape)
                max_strides = [
                    min(max_strides[0], max(1, image_shape[0])),
                    min(max_strides[1], max(1, image_shape[1])),
                ]
            elif lo == "Dropout":
                kwargs = self.chooseRandomComb(options["Dropout"])
            elif lo == "Flatten":
                kwargs = {}
            # elif l == "AveragePooling2D":
            #   pass
            else:
                print("Error: layer order contained unsupported layer: %s" % lo)
            kwargs_list.append(kwargs)
            new_layer_orders.append(lo)
            image_shape_list.append(image_shape.copy())

        kwargs = {}
        for k in ["Compile", "Fit"]:
            kwargs[k] = {}
            for item in options[k].keys():
                kwargs[k][item] = random.sample(options[k][item], 1)[0]
        kwargs_list.append(kwargs)
        return kwargs_list, new_layer_orders, image_shape_list
