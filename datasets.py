import os
import sys
import argparse
import ipdb
from glob import glob
import cv2
import copy
from tqdm import tqdm
import pandas
import re

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms

import numpy as np

import utils

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"





####
# UTIL FUNCTIONS
####
def folder_layers_walking(root_path, exclude_dirs=[]):
    data = []
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        if files != []:
            folder_layers = root.split(root_path)[-1]
            folder_layers = folder_layers.split('/')[1:]
            for file in files:
                if not (file.endswith(".h5") or file.endswith(".txt")):
                    data.append({
                        'img_path'      : os.path.join(root, file),
                        'folder_layers'  : folder_layers
                    })
    print("Sort data by alphabetical image path for consistency")
    data = sorted(data, key=lambda x: x['img_path'])
    return data




####
# Dataset Functions
####
def OM_splits(metadata):
    if metadata != ["instance"]:
        raise NotImplementedError(f"Finish implementing for anything other than just instance metadata")
    root_dir = "data/Durham_Oriental_Museum_Dataset"
    root_path = os.path.join(root_dir, "images")
    data = folder_layers_walking(root_path, exclude_dirs=["samian"])
    #xlsx = pandas.read_excel(os.path.join(root_dir, "Durham_University_Museums_data.xlsx"))
    #xlsx = xlsx.dropna(subset=["object_number"])
    ## remove non strings, and strings that have html elements in them
    #toremove = []
    #for idx, row in xlsx.iterrows():
    #    objn = row.object_number
    #    if type(objn) != str:
    #        toremove.append(idx)
    #    elif "<" in objn:
    #        toremove.append(idx)
    #    elif not(objn.startswith("EG") or objn.startswith("DUROM") or objn.startswith("DURMA") or objn.startswith("DURUC")):
    #        toremove.append(idx)
    #xlsx = xlsx.drop(toremove)
    splits = utils.split_container_by_ratio(data, (8,1,1))
    train_data = splits[0]
    valid_data = splits[1]
    test_data = splits[2]
    train = train_data
    valid = valid_data
    test = test_data
    class2label = {
        'castle':0,
        'egyptian':1,
        'fulling_mill':2,
        'oriental':3,
        'samian':4
    }
    metadata_dicts = {}
    if "class" in metadata:
        metadata_dicts["class"] = class2label 
    return train, valid, test, metadata_dicts



def OM_instances(train, valid, test, min_occ=3):
    instance2idx = {}
    occurrences = {}
    for idx, datum in enumerate(train):
        split_path = datum['img_path'].split("/")
        prefix = "/".join(split_path[3:-1])
        name = datum['img_path'].split("/")[-1]
        name = ".".join(name.split(".")[:-1])
        if split_path[3] == "egyptian":
            if name.lower().startswith("eg"):
                if name[2].isalnum():
                    name = name[2:]
                else:
                    name = name[3:]
            elif str(split_path[4]) in name:
                name = name.split(str(split_path[4]))[1]
                name = utils.remove_front_non_alnum(name)
            instance = utils.fname2instance_OM(name)
            if len(instance) > 0:
                if instance[-1].isalpha():
                    instance = instance[:-1]+instance[-1].lower()
            if instance != "":
                instance = "|".join(split_path[3:-1]+[instance])
        elif split_path[3] in ["castle","fulling_mill"]:
            if str(split_path[4]) in name:
                name = name.split(str(split_path[4]))[1]
                name = utils.remove_front_non_alnum(name)
                instance = utils.fname2instance_OM(name)
                if len(instance) > 0:
                    if instance[-1].isalpha():
                        instance = instance[:-1]+instance[-1].lower()
                # remove double letter at end
                if len(instance) > 2 and not(instance[-3].isalpha()):
                    if instance[-2:].isalpha():
                        instance = instance[:-2]
                        instance = utils.remove_end_non_alnum(instance)
                # FINALLY, FOR CASTLE AND FULLING MILL DATASET, SOMETIMES 5a 5b etc.. is the same, so we enforce this
                if len(instance) > 1:
                    if instance[-1].isalpha() and instance[-2].isnumeric():
                        instance = instance[:-1]
                        instance = utils.remove_end_non_alnum(instance)
                if instance != "":
                    instance = "|".join(split_path[3:-1]+[instance])
        elif split_path[3] == "oriental":
            if str(split_path[4]) in name:
                name = name.split(str(split_path[4]))[1]
                name = utils.remove_front_non_alnum(name)
            instance = utils.fname2instance_OM(name)
            if len(instance) > 2 and not(instance[-3].isalpha()):
                if instance[-2:].isalpha():
                    instance = instance[:-2]
                    instance = utils.remove_end_non_alnum(instance)
            if instance != "":
                instance = "|".join(split_path[3:-1]+[instance])
        else:
            print(split_path)
            raise NotImplementedError("The detected dataset is not supported!")
        # Handle instance
        #if instance == "":
        #    print("TRAIN", datum['img_path'])
        if instance not in instance2idx:
            instance2idx[instance] = len(instance2idx)
            occurrences[instance] = 1
        else:
            occurrences[instance] += 1
        train[idx]['instance'] = instance
    # Unseen instances in the validation / test splits are marked as unknown
    total = len(set(instance2idx.values())) # Extra unknown class
    for d_dict in [valid, test]:
        for idx, datum in enumerate(d_dict):
            split_path = datum['img_path'].split("/")
            prefix = "/".join(split_path[3:-1])
            name = datum['img_path'].split("/")[-1]
            name = ".".join(name.split(".")[:-1])
            if split_path[3] == "egyptian":
                if name.lower().startswith("eg"):
                    if name[2].isalnum():
                        name = name[2:]
                    else:
                        name = name[3:]
                elif str(split_path[4]) in name:
                    name = name.split(str(split_path[4]))[1]
                    name = utils.remove_front_non_alnum(name)
                instance = utils.fname2instance_OM(name)
                if len(instance) > 0:
                    if instance[-1].isalpha():
                        instance = instance[:-1]+instance[-1].lower()
                if instance != "":
                    instance = "|".join(split_path[3:-1]+[instance])
            elif split_path[3] in ["castle","fulling_mill"]:
                if str(split_path[4]) in name:
                    name = name.split(str(split_path[4]))[1]
                    name = utils.remove_front_non_alnum(name)
                    instance = utils.fname2instance_OM(name)
                    if len(instance) > 0:
                        if instance[-1].isalpha():
                            instance = instance[:-1]+instance[-1].lower()
                    # remove double letter at end
                    if len(instance) > 2 and not(instance[-3].isalpha()):
                        if instance[-2:].isalpha():
                            instance = instance[:-2]
                            instance = utils.remove_end_non_alnum(instance)
                    # FINALLY, FOR CASTLE AND FULLING MILL DATASET, SOMETIMES 5a 5b etc.. is the same, so we enforce this
                    if len(instance) > 1:
                        if instance[-1].isalpha() and instance[-2].isnumeric():
                            instance = instance[:-1]
                            instance = utils.remove_end_non_alnum(instance)
                    if instance != "":
                        instance = "|".join(split_path[3:-1]+[instance])
            elif split_path[3] == "oriental":
                if str(split_path[4]) in name:
                    name = name.split(str(split_path[4]))[1]
                    name = utils.remove_front_non_alnum(name)
                instance = utils.fname2instance_OM(name)
                if len(instance) > 2 and not(instance[-3].isalpha()):
                    if instance[-2:].isalpha():
                        instance = instance[:-2]
                        instance = utils.remove_end_non_alnum(instance)
                if instance != "":
                    instance = "|".join(split_path[3:-1]+[instance])
            else:
                print(split_path)
                raise NotImplementedError("The detected dataset is not supported!")
            # Handle instance
            #if instance == "":
            #    print("VALTEST", datum['img_path'])
            if instance not in instance2idx:
                instance2idx[instance] = total
                occurrences[instance] = 1
            else:
                occurrences[instance] += 1
            d_dict[idx]['instance'] = instance   
    # TODO Make a more elegant way to do this if its a bottleneck
    print("BECAUSE WE ARE REMOVING MINIMUM OCCURENCES, TRAIN VALID AND TEST SETS ARE RESHUFFLED")
    all_splits = train+valid+test
    enough_occurrences = []
    for datum in all_splits:
        if occurrences[datum['instance']] >= min_occ: # Cutoff for instances
            if datum['instance'] != "": # NOTE DO NOT COUNT ANY INSTANCE ENDS UP WITH AN EMPTY NAME UNDER THIS SCHEME
                enough_occurrences.append(datum)
    train, valid, test = utils.split_container_by_ratio(enough_occurrences, (8,1,1))
    instance2idx = {}
    for datum in train:
        instance = datum['instance']
        if instance not in instance2idx:
            instance2idx[instance] = len(instance2idx)
    total = len(set(instance2idx.values())) # Extra unknown class
    for d_dict in [valid, test]:
        for idx, datum in enumerate(d_dict):
            instance = datum['instance']
            if instance not in instance2idx:
                instance2idx[instance] = total
    occurrences = {k:v for k,v in occurrences.items() if v >= min_occ and k != ""}
    ###########################################################################
    #### Print dataset statistics
    imgs_len = sum(v for v in occurrences.values())
    instances_len = len(occurrences)
    print("\n\nTotal Dataset Length: ", imgs_len)
    print("Total Number of Instances: ", instances_len)
    
    imgs = []
    if min_occ <= 3:
        print("OCCURENCES == 3")
        occ_3 = sum(v==3 for v in occurrences.values())
        img_3 = 3*occ_3
        imgs.append(img_3)
        print("Instances: ", occ_3)
        print("% of Instances:" f"{100*occ_3/instances_len:.2f}%")
        print("Images: ", img_3)
        print("% of Images: ", f"{100*img_3/imgs_len:.2f}%")
        print("\n")

    if min_occ <= 4:
        print("OCCURENCES == 4")
        occ_4 = sum(v==4 for v in occurrences.values())
        img_4 = 4*occ_4
        imgs.append(img_4)
        print("Instances: ", occ_4)
        print("% of Instances:" f"{100*occ_4/instances_len:.2f}%")
        print("Images: ", img_4)
        print("% of Images: ", f"{100*img_4/imgs_len:.2f}%")
        print("\n")

    if min_occ <= 5:
        print("OCCURENCES == 5")
        occ_5 = sum(v==5 for v in occurrences.values())
        img_5 = 5*occ_5
        imgs.append(img_5)
        print("Instances: ", occ_5)
        print("% of Instances:" f"{100*occ_5/instances_len:.2f}%")
        print("Images: ", img_5)
        print("% of Images: ", f"{100*img_5/imgs_len:.2f}%")
        print("\n")

    if min_occ <= 6:
        print("OCCURENCES == 6")
        occ_6 = sum(v==6 for v in occurrences.values())
        img_6 = 6*occ_6
        imgs.append(img_6)
        print("Instances: ", occ_6)
        print("% of Instances:" f"{100*occ_6/instances_len:.2f}%")
        print("Images: ", img_6)
        print("% of Images: ", f"{100*img_6/imgs_len:.2f}%")
        print("\n")

    if min_occ <= 7:
        print("OCCURENCES == 7")
        occ_7 = sum(v==7 for v in occurrences.values())
        img_7 = 7*occ_7
        imgs.append(img_7)
        print("Instances: ", occ_7)
        print("% of Instances:" f"{100*occ_7/instances_len:.2f}%")
        print("Images: ", img_7)
        print("% of Images: ", f"{100*img_7/imgs_len:.2f}%")
        print("\n")

    if min_occ <= 8:
        print("OCCURENCES == 8")
        occ_8 = sum(v==8 for v in occurrences.values())
        img_8 = 8*occ_8
        imgs.append(img_8)
        print("Instances: ", occ_8)
        print("% of Instances:" f"{100*occ_8/instances_len:.2f}%")
        print("Images: ", img_8)
        print("% of Images: ", f"{100*img_8/imgs_len:.2f}%")
        print("\n")

    if min_occ > 8:
        raise NotImplementedError("Not supporting any runs with minimum occurences higher than 8")
    print("OCCURENCES == 9")
    occ_9 = sum(v==9 for v in occurrences.values())
    img_9 = 9*occ_9
    imgs.append(img_9)
    print("Instances: ", occ_9)
    print("% of Instances:" f"{100*occ_9/instances_len:.2f}%")
    print("Images: ", img_9)
    print("% of Images: ", f"{100*img_9/imgs_len:.2f}%")
    print("\n")

    print("OCCURENCES >= 10")
    occ_10 = sum(v>=10 for v in occurrences.values())
    img_10 = imgs_len-sum(imgs)
    print("Instances: ", occ_10)
    print("% of Instances:" f"{100*occ_10/instances_len:.2f}%")
    print("Images: ", img_10)
    print("% of Images: ", f"{100*img_10/imgs_len:.2f}%")
    print("\n")
    return train, valid, test, instance2idx, total, occurrences
    ###########################################################################

class GeneralDataset(Dataset):
    def __init__(self, dataset:str, shuffle:bool, metadata:list, transforms="resize", shuffle_random_seed=2667, min_occ=3, modelname=None):
        super().__init__()
        """
        metadata: metadata are passed in as unpacked arguments, i.e. *args
        """
        self.metadata = metadata
        # Transform
        if transforms == "resize":
            self.transforms = utils.resize_transforms()
        elif transforms == "resize-by-model":
            self.transforms = utils.resizebymodelonly_transforms(modelname)
        elif transforms == "resize-no-normalisation":
            self.transforms = utils.resizeonly_transforms()
        elif transforms == "none":
            self.transforms = torchvision.transforms.Compose([])
        else:
            raise NotImplementedError(f"Transform: '{args.transforms}' not recognised")

        # Create train, valid, test splits
        if dataset == "oriental-museum":
            self.train, self.valid, self.test, metadata_dicts = OM_splits(metadata)
        self.metadata_dicts = metadata_dicts

        # Shuffle the splits
        if shuffle:
            self.train, self.valid, self.test = utils.split_container_by_ratio(self.train+self.valid+self.test, (len(self.train), len(self.valid), len(self.test)), seed=shuffle_random_seed)

        # Instance2Idx
        if "instance" in metadata:
            if dataset == "oriental-museum":
                self.train, self.valid, self.test, instance2idx, total, occurrences = OM_instances(self.train, self.valid, self.test, min_occ=min_occ)
                self.metadata_dicts['occurrences'] = occurrences
            self.metadata_dicts['instance'] = instance2idx        # Instance2Idx
            self.total = total
            if self.total in self.metadata_dicts['instance'].values():
                if '@NULL_CLASS@' not in self.metadata_dicts['instance'].keys():
                    self.metadata_dicts['instance']['@NULL_CLASS@'] = self.total    # Add a null class if it does not already exists
                else:
                    raise ValueError("What are the chances that one of this instances in the dataset happen to have the SAME instance name as the token i want to reserve the null class if it doesn't exist")
        self.metadata_dicts = {key:value for key, value in self.metadata_dicts.items() if key in metadata+["occurrences"]}



    def choose_split(self, split):
        if split == "train":
            self.data = self.train
            del self.valid
            del self.test
        if split == "valid":
            self.data = self.valid
            del self.train
            del self.test
        if split == "test":
            self.data = self.test
            del self.train
            del self.valid



    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        data = self.data[idx]
        folder_layers = data['folder_layers']
        img_path = data['img_path']
        img = self.imgLoad(img_path)
        batch = {"img":img}
        if "instance" in self.metadata:
            batch['instance'] = int(self.metadata_dicts['instance'][data['instance']])
        if "class" in self.metadata:
            cls = self.metadata_dicts['class'][folder_layers[0]]
            batch['class'] = cls
        return batch



    def imgLoad(self, img_path):
        img = torch.from_numpy(cv2.imread(img_path, cv2.IMREAD_COLOR)).float()
        img = img.permute(2,0,1)    # C, H, W
        img = self.transforms(img)
        return img




class GeneralDatasetPreloaded(GeneralDataset):
    def __init__(self, dataset:str, shuffle:bool, metadata:list, transforms="resize", shuffle_random_seed=2667, min_occ=3):
        super().__init__(dataset, shuffle, metadata, transforms=transforms, shuffle_random_seed=shuffle_random_seed, min_occ=min_occ)

    def choose_split(self, split):
        self.all_data = []
        if split == "train":
            self.data = self.train
            del self.valid
            del self.test
        if split == "valid":
            self.data = self.valid
            del self.train
            del self.test
        if split == "test":
            self.data = self.test
            del self.train
            del self.valid
        for idx in tqdm(range(self.__len__()), total=self.__len__(), desc=f"Preloading {split} Dataset"):
            data = super().__getitem__(idx)
            self.all_data.append(data)

    def __getitem__(self, idx):
        return self.all_data[idx]


def resolutions_heatmap(resolutions_x, resolutions_y, bins):
    heatmap, xedges, yedges = np.histogram2d(resolutions_x, resolutions_y, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    cb = plt.colorbar()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

def resolutions_hexmap(resolutions_x, resolutions_y, gridsize=30, bins=400):
    n = 1e5
    #x = np.linspace(0, resolutions_x.max(), 300)
    #y = np.linspace(0, resolutions_y.max(), 300)
    x = resolutions_x
    y = resolutions_y
    X, Y = np.meshgrid(x, y)
    #x = X.ravel()
    #y = Y.ravel()
    z = None
    # if 'bins=None', then color of each hexagon corresponds directly to its count
    # 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then 
    # the result is a pure 2D histogram 
    plt.hexbin(x, y, C=z, gridsize=gridsize, cmap=CM.jet, bins=bins)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    #cb = plt.colorbar()
    #cb.set_label('mean value')
    plt.show()   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["oriental-museum"], default="oriental-museum", help="""
    args.dataset
    """)
    parser.add_argument("--metadata", type=str, nargs='+', help="""
    List of strings for metadata for ML
    args.metadata
        class = The `classes' of the images. Classes vary between datasets. See the above functions for details
        instance = The instance of image (there may be multiple images of the same instance)
    """)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")
    parser.add_argument("--preload", action="store_true", help="preload dataset. Run this when dataset loading is a bottleneck")
    parser.add_argument("--bsz", default=1, type=int, help="Training batch size")
    parser.add_argument("--vt_bsz", default=1, type=int, help="Validation and test batch size")
    parser.add_argument("--num_workers", default=0, type=int, help="number of pytorch workers for dataloader")
    parser.add_argument("--transforms", type=str, choices=["augment", "resize", "resize-by-model", "resize-no-normalisation", "none"], default="resize", help="""
    The kind of transforms to use in preprocessing for images
    args.transforms
        resize = Only crop down and resize
        none = No transforms
        augment = COMBINE (CONCATENATE) a dataset using just resize with a dataset also use transforms (i.e. data augmentation)
    """)
    parser.add_argument("--min_occ", type=int, default=6, help="Minimum number of occurences required to keep the class for instances")
    parser.add_argument("--dset_seed", type=int, default=2667, help="Random seed to make the dataset and fc_intermediate consistent")

    # NOTE 
    # FOR DAN
    # camille: --dataset camille --metadata instance

    args = parser.parse_args()
    if args.metadata == None:
        args.metadata = []

    # Datasets
    dataset_switch = {0:GeneralDataset, 1:GeneralDatasetPreloaded}  # args.preload to force a preloading of all images from potentially faster training
    if args.transforms == "augment":    # args.transforms == "augment" means combining a dataset without transforms, and one with, using ConcatDataset
        train_dsets = []    # Lists from which to concatenate
        valid_dsets = []
        test_dsets = []
        # Original dataset 
        ## To ensure correct splitting of train/valid/test, datasets call a specific "choose_split" function as seen below, requiring copying
        original_dataset = dataset_switch[args.preload](args.dataset, args.shuffle, args.metadata, transforms="resize", min_occ=args.min_occ, modelname=args.model)
        valid_data = copy.deepcopy(original_dataset)
        valid_data.choose_split("valid")
        valid_dsets.append(valid_data)
        test_data = copy.deepcopy(original_dataset)
        test_data.choose_split("test")
        test_dsets.append(test_data)
        train_data = original_dataset
        train_data.choose_split("train")
        train_dsets.append(train_data)
        metadata_dicts = train_data.metadata_dicts  # Get the dictionarys of metadata which contain for example, instance2idx dictionary
        # Transform dataset
        transform_dataset = dataset_switch[args.preload](args.dataset, args.shuffle, args.metadata, transforms=args.transforms, min_occ=args.min_occ, modelname=args.model)
        valid_data = copy.deepcopy(transform_dataset)
        valid_data.choose_split("valid")
        valid_dsets.append(valid_data)
        test_data = copy.deepcopy(transform_dataset)
        test_data.choose_split("test")
        test_dsets.append(test_data)
        train_data = transform_dataset
        train_data.choose_split("train")
        train_dsets.append(train_data)
        train_data = torch.utils.data.ConcatDataset(train_dsets)
        valid_data = torch.utils.data.ConcatDataset(valid_dsets)
        test_data = torch.utils.data.ConcatDataset(test_dsets)
    else:
        data = dataset_switch[args.preload](args.dataset, args.shuffle, args.metadata, transforms=args.transforms, min_occ=args.min_occ, modelname=None)
        valid_data = copy.deepcopy(data)
        valid_data.choose_split("valid")
        test_data = copy.deepcopy(data)
        test_data.choose_split("test")
        train_data = data
        train_data.choose_split("train")
        metadata_dicts = train_data.metadata_dicts
    # Dataloader
    train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=1)
    valid_loader = DataLoader(valid_data, num_workers=args.num_workers, batch_size=1)
    test_loader = DataLoader(test_data, num_workers=4, batch_size=1)#args.vt_bsz)

    # Iterate across the dataset
    # print([k for k in metadata_dicts['instance'].keys() if not(k.startswith("castle") or k.startswith("egyptian") or k.startswith("oriental") or k.startswith("fulling_mill"))])
    # print("")
    import matplotlib.pyplot as plt
    from matplotlib import cm as CM
    from matplotlib import mlab as ML
    import matplotlib.patches as patches
    import seaborn as sns
    #sns.set()
    #fig, ax = plt.subplots()
    resolutions_x = []
    resolutions_y = []
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        resolutions_x.append(batch['img'].shape[2])
        resolutions_y.append(batch['img'].shape[3])
    for idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        resolutions_x.append(batch['img'].shape[2])
        resolutions_y.append(batch['img'].shape[3])
    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        resolutions_x.append(batch['img'].shape[2])
        resolutions_y.append(batch['img'].shape[3])
    resolutions_x = np.asarray(resolutions_x)
    resolutions_y = np.asarray(resolutions_y)
   
    # The ticks
    #ax.xaxis.set_tick_params(width=3)
    #ax.yaxis.set_tick_params(width=3)
    #plt.grid()
    #fig.align_labels()
    # keep only images with resolution <= (1000, 1000)
    to_remove = np.argwhere(np.logical_and(resolutions_x <= 1000, resolutions_y <= 1000) == False)
    in_x, in_y = np.delete(resolutions_x, to_remove), np.delete(resolutions_y, to_remove)
    #ax.set_xlim(0, 1000)
    #ax.set_ylim(0, 1000)
    # Jointplot
    sjp = sns.jointplot(x=in_x, y=in_y, kind='scatter', marker="x", xlim=(0,1000), ylim=(0,1000), color="black")
    #breakpoint()
    plt.suptitle('Resolutions of Images in the Full Dataset', fontweight="bold", y=0.02)
    sjp.ax_joint.set_xlabel("Width (Pixels)", fontweight="bold")
    sjp.ax_joint.set_ylabel('Height (Pixels)', fontweight="bold")
 
    sjp.ax_joint.plot(0,0, color="#e80911", label="EfficientNet-b[0-7]")
    sjp.ax_joint.plot(0,0, color="#0000ff", label="Inception-v3/4")
    sjp.ax_joint.plot(0,0, color="#57ad6e", label="ResNet Rescaled")
    # ENET
    sjp.ax_joint.plot([256,256],[0  ,224], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([0  ,256],[224,224], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([256,256],[0  ,240], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([0  ,256],[240,240], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([288,288],[0  ,288], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([0  ,288],[288,288], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([320,320],[0  ,300], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([0  ,320],[300,300], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([384,384],[0  ,380], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([0  ,384],[380,380], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([489,489],[0  ,456], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([0  ,489],[456,456], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([561,561],[0  ,528], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([0  ,561],[528,528], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([633,633],[0  ,600], color="#e80911", linewidth=1)
    sjp.ax_joint.plot([0  ,633],[600,600], color="#e80911", linewidth=1)
    # Inception
    sjp.ax_joint.plot([299,299],[0  ,299], color="#0000ff", linewidth=1)
    sjp.ax_joint.plot([0  ,299],[299,299], color="#0000ff", linewidth=1)
    # ResNet Rescaled
    sjp.ax_joint.plot([224,224],[0  ,224], color="#57ad6e", linewidth=1)
    sjp.ax_joint.plot([0  ,224],[224,224], color="#57ad6e", linewidth=1)
    ##
    # rectangles
    #"enetb0":(256,224),
    #"enetb1":(256,240),
    #"enetb2":(288,288),
    #"enetb3":(320,300),
    #"enetb4":(384,380),
    #"enetb5":(489,456),
    #"enetb6":(561,528),
    #"enetb7":(633,600),
    #"inceptionv3":(299,299),
    #"inceptionv4":(299,299),
    #"resnet_rs_101":(224,224)
    sjp.ax_joint.legend(loc="best")
    plt.show()
    #resolutions_hexmap(in_x, in_y, 100, bins=400)
    #resolutions_heatmap(in_x, in_y, (1000,1000))

