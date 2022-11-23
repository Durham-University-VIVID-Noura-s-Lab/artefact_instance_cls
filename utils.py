import numpy as np
import pickle
from string import ascii_lowercase, ascii_uppercase
import re

import torchvision.transforms as transforms

tosplit = [i+i for i in ascii_uppercase]
tosplit += [i+i for i in ascii_lowercase]
tosplit += ["q"+str(i) for i in range(1,10)]
tosplit += ["q"+f"{i:02}" for i in range(0,20)]
tosplit += ["d"+str(i) for i in range(1,10)]
tosplit += ["d"+f"{i:02}" for i in range(0,20)]
tosplit += ["pc","_small","cu"]

tosplit_cond = "|".join(tosplit)

def remove_front_non_alnum(string):
    if string == "":
        return string
    while not(string[0].isalnum()):
        string = string[1:]
    return string

def remove_end_non_alnum(string):
    if string == "":
        return string
    while not(string[-1].isalnum()):
        string = string[:-1]
    return string

def fname2instance_OM(fname):
    instance = re.split(tosplit_cond, fname)
    instance = instance[0] #"".join(instance)
    instance = remove_end_non_alnum(instance)
    return instance 

def load_pickle(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def split_container_by_ratio(data, split_ratio, seed=2667):
    """
    Split a python container e.g. list, dataset, into random splits
        - data = python container
        - split_ratio = (1,2,3) as a tuple of ratios to split
        - seed = random fixable seed
    """
    # Fixed random shuffling
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(data)
    splits = []
    total = sum(split_ratio)
    for idx in range(len(split_ratio)):
        start = sum(split_ratio[:idx])/total
        start = round(start*len(data))
        end = sum(split_ratio[:idx+1])/total
        end = round(end*len(data))
        splits.append(data[start:end])
    return splits

def resizeonly_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
    ])

def resizebymodelonly_transforms(modelname):
    resize_switch = {
        "enetb0":(256,224),
        "enetb1":(256,240),
        "enetb2":(288,288),
        "enetb3":(320,300),
        "enetb4":(384,380),
        "enetb5":(489,456),
        #"enetb6":(384,380),
        #"enetb7":(384,380),
        "enetb6":(561,528),    # Original
        "enetb7":(633,600),    # Original
        "inceptionv3":(299,299),
        "inceptionv4":(299,299),
        "resnet_rs_101":(224,224)
    }
    return transforms.Compose([
        transforms.Resize(resize_switch[modelname]),
    ])

def resize_transforms():
    return transforms.Compose([
        # Normalise
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
        transforms.Resize((224,224)),
    ])
