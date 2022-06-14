from tools.hardware.gpu_feature import GpuFeatures


def get_gpu_features():
    # need to run with GPU
    # only with default features
    gpufeature = GpuFeatures()
    print(gpufeature.get_features())

    # with additional features
    gpufeature = GpuFeatures(
        features=["power.management", "power.limit"], with_dafault_features=True
    )
    print(gpufeature.get_features())

    # only use new features
    gpufeature = GpuFeatures(
        features=["power.management", "power.limit"], with_dafault_features=False
    )
    print(gpufeature.get_features())


if __name__ == "__main__":
    get_gpu_features()
