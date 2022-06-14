import os
import re
import subprocess as sp


class GeneralFeatures:
    def __init__(self, features=None, with_dafault_features=True, install_lshw=False):
        """
        if doesn't work run "sudo apt-get install lshw" first in command
        check here or lshw documentation for all class names:
        https://myl1nux.wordpress.com/2010/06/02/how-to-get-hardware-info-in-linux/

        @param features:
        @param with_dafault_features:
        """
        if install_lshw:
            # TODO: Support more OS
            os.system("sudo apt-get install lshw")
        self.default_features = ["cpu", "memory", "network"]
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
            r"[ ]{2,10}\*-", output
        )  # could be {2, any} # put something big is find
        sub_values = []

        for sep in separators:
            pos = output.index(sep)
            sub_values.append(output[:pos])
            output = output[pos + len(sep) :]
        sub_values.append(output)

        sub_values_data = []
        for sub_value in sub_values:
            new = re.sub("[ ]{2,}", "", sub_value)
            data = dict(
                tuple(pair.split(": "))
                for pair in new[new.find("\n") + 1 :].split("\n")
                if pair
            )
            name = new[: new.find("\n")]
            name_data = {name: data}
            sub_values_data.append(name_data)
        return sub_values_data

    def get_features(self):
        feature_dict = dict()
        for feature in self.features:
            output = sp.getoutput(f"sudo lshw -class {feature}")
            parsed_output = self.get_sub_values(output)
            for o in parsed_output:
                for i, k in enumerate(o):
                    new_subkeys = [
                        f"{k}_" + "_".join(subkey.split()) for subkey in o[k]
                    ]
                    tmp_feature_dict = dict(zip(new_subkeys, list(o[k].values())))
                    feature_dict.update(tmp_feature_dict)
        # print(feature_dict)
        return feature_dict
