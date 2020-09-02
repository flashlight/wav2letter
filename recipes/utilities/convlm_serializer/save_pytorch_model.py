from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from collections import defaultdict

import torch


def convert(model_state, key, suffix=""):
    string = ""
    param = model_state[key]

    # param name
    string += ".".join(key.split(".")[1:-1]) + suffix + "." + key.split(".")[-1] + " "
    change_to_lin_layer = False
    if "conv" in key and len(param.shape) == 3:
        if ("weight_v" in key and param.shape[0] == 1) or (
            "weight_g" in key
            and model_state[key.replace("weight_g", "weight_v")].shape[0] == 1
        ):
            change_to_lin_layer = True
    if change_to_lin_layer:
        # param shapes
        string += (
            str(len(param.shape) - 1) + " " + " ".join(map(str, param.shape[1:][::-1]))
        )
        # param matrix
        string += " " + " ".join(map(str, param.cpu().numpy()[0].T.flatten()))
    else:
        # param shapes
        string += str(len(param.shape)) + " " + " ".join(map(str, param.shape))
        # param matrix
        string += " " + " ".join(map(str, param.cpu().numpy().flatten()))
    return string


def save_model(pytorch_model_path, dst):
    model_state = torch.load(pytorch_model_path)
    model_state = model_state["model"]
    add_string = ""
    prev_key = ""

    with open(dst, "w") as f:
        projections = defaultdict(list)
        for key in model_state:
            print("Process param", key)
            if "version" in key:
                print("Skip", key)
                continue
            if "projection" in key:
                projections[key.split(".")[-2]].append(
                    convert(model_state, key, "-projection")
                )
            else:
                if prev_key != key.split(".")[2]:
                    if add_string != "":
                        f.write(add_string + "\n")
                    add_string = ""
                prev_key = key.split(".")[2]
                if key.split(".")[2] in projections:
                    add_string = "\n".join(projections[key.split(".")[2]])
                f.write(convert(model_state, key) + "\n")


if __name__ == "__main__":
    print("Converting the model. Usage: save_pytorch_model.py [path/to/model] [dst]")
    path = sys.argv[1]
    dst = sys.argv[2]
    save_model(path, dst)
