"""
Use this link for full more reference for gpu_features
https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
"""
import subprocess as sp


class GpuFeatures:
    def __init__(self, features=None, with_dafault_features=True):
        """
        use "nvidia-smi --help-query-gpu" to see all available features

        @param features: None or a list of name of features(str), full list can be found with "nvidia-smi --help-query-gpu"
        @param with_dafault_features: if return new added features wit default features
        """
        self.default_features = ",".join(
            [
                "timestamp",
                "driver_version",
                "count",
                "gpu_name",
                "pcie.link.width.max",
                "vbios_version",
                "memory.total,temperature.gpu",
            ]
        )
        self.features = None
        if features is None:
            self.features = self.default_features
        else:
            if with_dafault_features:
                self.features = self.default_features
                for feature in features:
                    if feature in self.default_features:
                        pass
                    else:
                        self.features += "," + feature
            else:
                self.features = ",".join(features)

    def get_features(self):
        output = sp.getoutput(
            f"nvidia-smi --query-gpu={self.features} --format=csv --format=csv"
        )
        output = output.replace(", ", ",")
        keys = [f"gpu_{o}" for o in output.split("\n")[0].split(",")]
        values = output.split("\n")[1].split(",")
        return dict(zip(keys, values))
