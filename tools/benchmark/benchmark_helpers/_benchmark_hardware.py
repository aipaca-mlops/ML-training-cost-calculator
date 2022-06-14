def _grab_general_hardware_features(self, key="general"):
    from tools.hardware.general_feature import GeneralFeatures

    gfeature = GeneralFeatures()
    features = self._filter_feature(gfeature.get_features(), key)
    return features


def _grab_gpu_features(self, key="gpu"):
    from tools.hardware.gpu_feature import GpuFeatures

    gpu_features = GpuFeatures()
    features = self._filter_feature(gpu_features.get_features(), key)
    return features


def grab_hardware_features(self, dump=True, key="h"):
    hardware_feature = self._merge_dicts(
        self._grab_general_hardware_features(), self._grab_gpu_features()
    )
    self.hardware_id = self._gen_id(hardware_feature, key)
    hardware_features = [
        self._merge_dicts({"hardware_id": self.hardware_id}, hardware_feature)
    ]
    if dump:
        self._write_hardware_csv(hardware_features)
