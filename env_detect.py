import subprocess as sp
import os
import re
"""
Use this link for full more reference for gpu_features
https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
"""


class gpu_features:
    def __init__(self, features=None, with_dafault_features=True):
        """
        use "nvidia-smi --help-query-gpu" to see all available features

        @param features: None or a list of name of features(str), full list can be found with "nvidia-smi --help-query-gpu"
        @param with_dafault_features: if return new added features wit default features
        """
        self.default_features = 'timestamp,driver_version,count,gpu_name,pcie.link.width.max,vbios_version,memory.total,temperature.gpu'
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
                        self.features += ',' + feature
            else:
                self.features = ','.join(features)

    def get_features(self):
        output = sp.getoutput(f'nvidia-smi --query-gpu={self.features} --format=csv --format=csv')
        output = output.replace(', ', ',')
        return dict(zip(output.split('\n')[0].split(','), output.split('\n')[1].split(',')))


class general_features:
    def __init__(self, features=None, with_dafault_features=True, install_lshw=False):
        """
        if doesn't work run "sudo apt-get install lshw" first in command
        check here or lshw documentation for all class names:
        https://myl1nux.wordpress.com/2010/06/02/how-to-get-hardware-info-in-linux/

        @param features:
        @param with_dafault_features:
        """
        if install_lshw:
            os.system("sudo apt-get install lshw")
        self.default_features = ['cpu', 'memory', 'network']
        self.features = None
        if features is None:
            self.features = self.default_features.copy()
        else:
            if with_dafault_features:
                self.features = self.default_features.copy()
                for feature in features:
                    if feature in self.default_features:
                        pass
                    else:
                        self.features.append(feature)
            else:
                self.features = features

    def get_sub_values(self, output):
        output = output[4:]
        separators = re.findall(
            '[ ]{2,10}\*-', output
        )  # could be {2, any} # put something big is find
        sub_values = []

        for sep in separators:
            pos = output.index(sep)

            sub_values.append(output[:pos])
            output = output[pos + len(sep):]
        sub_values.append(output)

        sub_values_data = []
        for sub_value in sub_values:
            new = re.sub('[ ]{2,}', '', sub_value)
            data = dict(
                tuple(pair.split(': ')) for pair in new[new.find('\n') + 1:].split('\n') if pair
            )
            name = new[:new.find('\n')]
            name_data = {name: data}
            sub_values_data.append(name_data)
        return sub_values_data

    def get_features(self):
        feature_dict = dict()
        for feature in self.features:
            output = sp.getoutput(f'sudo lshw -class {feature}')
            feature_dict[feature] = self.get_sub_values(output)
        return feature_dict


def get_general_features():
    # only with default features
    gfeature = general_features()
    print(gfeature.get_features())

    # with additional features
    gfeature = general_features(features=['disk', 'volume'], with_dafault_features=True)
    print(gfeature.get_features())

    # only use new features
    gfeature = general_features(features=['disk', 'volume'], with_dafault_features=False)
    print(gfeature.get_features())


def get_gpu_features_demo():
    # need to run with GPU
    # only with default features
    gpufeature = gpu_features()
    print(gpufeature.get_features())

    # with additional features
    gpufeature = gpu_features(
        features=['power.management', 'power.limit'], with_dafault_features=True
    )
    print(gpufeature.get_features())

    # only use new features
    gpufeature = gpu_features(
        features=['power.management', 'power.limit'], with_dafault_features=False
    )
    print(gpufeature.get_features())
