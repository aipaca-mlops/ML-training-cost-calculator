from tools.hardware.general_feature import GeneralFeatures


def get_general_features():
    # only with default features
    gfeature = GeneralFeatures()
    print(gfeature.get_features())

    # with additional features
    gfeature = GeneralFeatures(features=["disk", "volume"], with_dafault_features=True)
    print(gfeature.get_features())

    # only use new features
    gfeature = GeneralFeatures(features=["disk", "volume"], with_dafault_features=False)
    print(gfeature.get_features())


if __name__ == "__main__":
    get_general_features()
