__author__ = "jumperkables"
import os, sys
from tqdm import tqdm

# TODO REMOVE ME

# Transforms
def transform_img(img_path):
    return tran1_img, tran1_img_path, tran2_img, tran2_img_path



root_path = "/home/jumperkables/archaeology/data/Durham_Oriental_Museum_Dataset/images_augment"
exclude_dirs = ["samian"]
data = []
for root, dirs, files in os.walk(root_path):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    if files != []:
        for file in files:
            if not (file.endswith(".h5") or file.endswith(".txt")):
                data.append({
                    'img_path'      : os.path.join(root, file),
                })


# Every image path is in data
for i_dict in tqdm(data, total=len(data)):
    img_path = i_dict['img_path']

