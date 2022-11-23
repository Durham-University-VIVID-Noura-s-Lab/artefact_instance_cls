import os
import sys
import argparse
import ipdb
from glob import glob
import cv2
import copy
from tqdm import tqdm
import pandas
import shutil
import re

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms

import numpy as np

import utils

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

prefixes = []
for root, dirs, files in os.walk("data/Durham_Oriental_Museum_Dataset/images"):
    dirs[:] = [d for d in dirs if d not in []]
    #### REMOVE THE YEAR
    #if files != []:
    #    for file in files:
    #        prefix = re.split("\W+", file)[0]
    #        if prefix != "":
    #            remove=root.split("/")[-1]
    #            if file.startswith(remove):
    #                if file[len(remove):].isalnum():
    #                    breakpoint()
    #                    new_fname = file[len(remove):]
    #                else:
    #                    new_fname = file[len(remove)+1:]
    #                os.rename(os.path.join(root, file), os.path.join(root, new_fname))

    #### REMOVE PREFIXES
    if files != []:
        for file in files:
            prefix = re.split("\W+", file)[0]
            if prefix != "":
                if not prefix[0].isnumeric():
                    remove='DURUC'
                    if file.startswith(remove):
                        if file[len(remove):].isalnum():
                            breakpoint()
                            new_fname = file[len(remove):]
                        else:
                            breakpoint()
                            new_fname = file[len(remove)+1:]
                        os.rename(os.path.join(root, file), os.path.join(root, new_fname))
                    prefixes.append(os.path.join(root,file))
print(set(prefixes))
print(len(set(prefixes)))

    #### empty recursive subfolders
    #if files != []:
    #    for file in files:
    #        layers = root.split("/")
    #        if len(layers) > 5:
    #            fname = os.path.join(root, file)
    #            new_loc = "/".join(layers[:5])
    #            shutil.move(fname, os.path.join(new_loc, file))

    #### Remove Prefix
    #if files != []:
    #    for file in files:
    #        # Lower the file extension
    #        #breakpoint()
    #        if file.startswith("EG"):
    #            new_fname = file[2:]
    #            os.rename(os.path.join(root, file), os.path.join(root, new_fname))

    ## LOWER FILE EXTENSIONS
    #if files != []:
    #    for file in files:
    #        # Lower the file extension
    #        #breakpoint()
    #        ext = file.split(".")[-1].lower()
    #        new_fname = ".".join(file.split(".")[:-1])
    #        new_fname = f"{new_fname}.{ext}"
    #        os.rename(os.path.join(root, file), os.path.join(root, new_fname))
