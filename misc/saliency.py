import os, sys
import argparse
import copy
from tqdm import tqdm
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
  
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
from datasets import GeneralDataset
from main import LMSystem

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import timm

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from captum.attr import *

def only_calc_attribution(img, target, method, method_name):
    if method_name == "SHAP":
        attribution = method.attribute(img, baselines=torch.zeros(img.shape, device=img.device), target=target)
    else:
        attribution = method.attribute(img, target=target)
    img = img.cpu().numpy()[0]
    attribution = attribution.cpu().numpy()[0]
    img = np.transpose(img, (1,2,0))
    attribution = np.transpose(attribution, (1,2,0))
    return attribution


def visualise_method(img, target, save_path, method, method_name):
    if os.path.exists(save_path):
        os.remove(save_path)
    if method_name == "SHAP":
        attribution = method.attribute(img, baselines=torch.zeros(img.shape, device=img.device), target=target)
    else:
        attribution = method.attribute(img, target=target)
    img = img.cpu().numpy()[0]
    attribution = attribution.cpu().numpy()[0]
    img = np.transpose(img, (1,2,0))
    attribution = np.transpose(attribution, (1,2,0))
    fig = visualization.visualize_image_attr(attr=attribution, alpha_overlay=0.5, original_image=img, method="blended_heat_map", use_pyplot=False, cmap="YlOrRd") # This flag as false will allow saving
    fig[0].savefig(save_path)


class EnsembleForSaliencyMemorySave(nn.Module):
    def __init__(self, all_models, device):
        super(EnsembleForSaliencyMemorySave, self).__init__()
        self.all_models = all_models
        self.device = device
        self.model_idx = 0

    def forward(self, x):
        model = self.all_models[self.model_idx]
        mod_output = model(x)
        mod_output = torch.softmax(mod_output, dim=1)
        self.model_idx += 1
        self.model_idx = self.model_idx % len(self.all_models)
        return(mod_output)


class EnsembleForSaliency(nn.Module):
    def __init__(self, all_models, device):
        super(EnsembleForSaliency, self).__init__()
        self.all_models = all_models
        self.device = device

    def forward(self, x):
        outputs = []
        for model in self.all_models:
            mod_output = model(x)
            outputs.append(mod_output)
        vote_out = torch.zeros(outputs[0].shape).to(self.device)
        #vote_out = vote_out.to(self.device)
        for o in outputs:
            vote_out += torch.softmax(o, dim=1)
        vote_out = torch.softmax(vote_out, dim=1)
        return(vote_out)


class ModelForSaliency(nn.Module):
    def __init__(self, encoder, fc):
        super(ModelForSaliency, self).__init__()
        self.encoder = encoder
        self.fc = fc
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return(x)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", choices=["vgg16","vgg19","resnet18","resnet34","resnet50","resnet101","resnet152","densenet161","resnet_rs_101", "san", "enetb0","enetb1","enetb2","enetb3","enetb4","enetb5","enetb6","enetb7","inceptionv3","inceptionv4"], required=True, type=str, help="model to use")
    parser.add_argument("--num_workers", default=0, type=int, help="number of pytorch workers for dataloader")
    parser.add_argument("--device", default=0, type=int, help="Cuda device, -1 for CPU")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle the dataset")
    parser.add_argument("--encoder_freeze", type=int, default=0, help="Freeze the CNN, SAN or other substantial part of the network, i.e. things that aren't the fully connected classifier. 0=dont freeze, 1=freeze")
    parser.add_argument("--dropout", type=float, default=0.2, help="FC dropout")
    parser.add_argument("--fc_intermediate", type=int, default=512, help="Intermediate size of the FC layers")
    parser.add_argument("--dataset", type=str, choices=["oriental-museum"], default="thesis", help="""
    args.dataset
        gallica_coin = split gallica coins dataset gathered by Tom Winterbottom
    """)
    parser.add_argument("--transforms", type=str, choices=["augment", "resize", "none","resize-by-model"], default="none", help="""
    The kind of transforms to use in preprocessing for images
    args.transforms
        resize = Only crop down and resize
        none = No transforms
        augment = COMBINE (CONCATENATE) a dataset using just resize with a dataset also use transforms (i.e. data augmentation)
    """)
    parser.add_argument("--metadata", type=str, nargs='+', help="""
    List of strings for metadata for ML
    args.metadata
        class = The `classes' of the images. Classes vary between datasets. See the above functions for details
        instance = The instance of image (there may be multiple images of the same instance)
    """)
    parser.add_argument("--dset_seed", type=int, default=2667, help="Random seed to make the dataset and fc_intermediate consistent")
    parser.add_argument("--min_occ", type=int, default=3, help="Minimum number of occurences required to keep the class for instances")
    parser.add_argument("--test_ckpt_path", nargs="+", type=str, default="", help="If specified, load the ckpt in and run it through the test set")
    parser.add_argument("--loss_weight_scaling", action="store_true", help="Activate this flag to enable weighted loss scaling for the multiclass problems")
    parser.add_argument("--feats_gen", action="store_true", help="If activated, store the final feature vectors generated before the linear layer (for visualisation purposes)")
    parser.add_argument("--ensemble", action="store_true", help="Run a test-only ensemble of models")
    parser.add_argument("--method", required=True, choices=["saliency", "IG", "SHAP"], help="Which method of saliency-like analysis to run")
    parser.add_argument("--save_ensemble_memory", action="store_true", help="Process the votes of each model one at a time and average (ensemble models have huge memory footprints)")
    parser.add_argument("--conf_matrix", action="store_true", help="If activated, store the final feature vectors generated before the linear layer (for visualisation purposes)")
    # IG == Integrated gradients
    args = parser.parse_args()

    # Ensemble argument checking
    if not args.ensemble:
        assert len(args.model) == 1, f"When not in ensmeble mode, only a single model should be supplied"
        args.model = args.model[0]
        if args.test_ckpt_path != "":
            args.test_ckpt_path = args.test_ckpt_path[0]
    else:
        print("\n\nWHEN ENSEMBLING, REMEMBER THAT args.dset_seed and args.min_occ SHOULD BE CONSISTENT, OTHERWISE THE INSTANCE INDEXING WILL NOT BE ALIGNED\n\n")
        assert len(args.model) == len(args.test_ckpt_path), f"If ensembling, the number of models specified should of course be matched by the number of test_ckpt_path"

    # Create jobname for logging and saving
    min_occ_str = f"-minocc={args.min_occ}" if "instance" in args.metadata else ""
    if args.metadata == None:   # args.metadata holds a list of all metadata for classification
        args.metadata = []

    # Datasets
    if args.transforms == "augment":    # args.transforms == "augment" means combining a dataset without transforms, and one with, using ConcatDataset
        train_dsets = []    # Lists from which to concatenate
        valid_dsets = []
        test_dsets = []
        # Original dataset 
        ## To ensure correct splitting of train/valid/test, datasets call a specific "choose_split" function as seen below, requiring copying
        if len(args.model) != 1:
            raise NotImplementedError("Fix the modelname parameter on the line below for multiple models again")
        original_dataset = GeneralDataset(args.dataset, args.shuffle, args.metadata, transforms="resize", min_occ=args.min_occ, modelname=args.model[0])
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
        if len(args.model) != 1:
            raise NotImplementedError("Fix the modelname parameter on the line below for multiple models again")
        transform_dataset = GeneralDataset(args.dataset, args.shuffle, args.metadata, transforms=args.transforms, min_occ=args.min_occ, modelname=args.model[0])
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
        if len(args.model) != 1:
            raise NotImplementedError("Fix the modelname parameter on the line below for multiple models again")
        data = GeneralDataset(args.dataset, args.shuffle, args.metadata, transforms=args.transforms, min_occ=args.min_occ, modelname=args.model[0])
        valid_data = copy.deepcopy(data)
        valid_data.choose_split("valid")
        test_data = copy.deepcopy(data)
        test_data.choose_split("test")
        train_data = data
        train_data.choose_split("train")
        metadata_dicts = train_data.metadata_dicts
    # Dataloader
    #train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.bsz)
    #valid_loader = DataLoader(valid_data, num_workers=args.num_workers, batch_size=args.vt_bsz)
    test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=1)#args.vt_bsz)
    #breakpoint()
    #sys.exit()
    # GPU
    if args.device == -1:
        gpus = None
        args.device = "cpu"
    else: 
        print("Add support for multiple GPUs")
        gpus = [args.device]


    if len(args.metadata) > 1:
        # Where usually we classify multiple metadata at the same time, i have not continued support for it recently, and am thus raising this error
        raise NotImplementedError("Fix the callbacks to pick what to monitor in the case of multiple metadata classification")

    assert args.test_ckpt_path != "", f"Should only be ran on model checkpoints"
    all_models = []
    for idx in range(len(args.model)):
        copy_args = copy.deepcopy(args)
        copy_args.model = args.model[idx]
        args.test_ckpt_path[idx] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.test_ckpt_path[idx]))
        copy_args.test_ckpt_path = args.test_ckpt_path[idx]
        pl_system = LMSystem.load_from_checkpoint(args.test_ckpt_path[idx], args=copy_args, metadata_dicts=copy.deepcopy(metadata_dicts))
        net = ModelForSaliency(pl_system.encoder,pl_system.fc)
        net.eval()
        net.to(args.device)
        all_models.append(net)
    if args.save_ensemble_memory:
        net = EnsembleForSaliencyMemorySave(all_models, args.device)
    else:
        net = EnsembleForSaliency(all_models, args.device)
    net.eval()
    net = net.to(args.device)
    if args.method == "saliency":
        method = Saliency(net)
    elif args.method == "IG":
        method = IntegratedGradients(net)
    elif args.method == "SHAP":
        method = GradientShap(net)
    # Split all models and votes up to save memory
    if args.save_ensemble_memory:
         for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            instance = int(batch['instance'][0])
            img = batch['img']
            img = img.to(args.device)
            pred = net(img)
            pred = int(pred.argmax())
            if pred == instance:
                if len(args.model) != 1:
                    raise NotImplementedError("Fix the fstring below to not just take the first model")
                save_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../.{args.method}/min_occ{args.min_occ}-{args.model[0]}-{idx}-correct-{instance}.png")) 
                total_attribution = np.zeros(img[0].permute(1,2,0).shape)
                for _ in range(len(args.model)):
                    #torch.cuda.empty_cache()
                    attribution = only_calc_attribution(img, instance, method, args.method)
                    total_attribution += attribution
                total_attribution = total_attribution / float(len(args.model))
                img = img.cpu().numpy()[0]
                img = np.transpose(img, (1,2,0))
                fig = visualization.visualize_image_attr(attr=total_attribution, original_image=img, method="blended_heat_map", use_pyplot=False, cmap="YlOrRd") # This flag as false will allow saving
            else:
                # INCORRECT ONE
                if len(args.model) != 1:
                    raise NotImplementedError("Fix the fstring below to not just take the first model")
                save_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../.{args.method}/min_occ{args.min_occ}-{args.model[0]}-{idx}-wrong-{instance}-{pred}.png"))
                total_attribution = np.zeros(img[0].permute(1,2,0).shape)
                for _ in range(len(args.model)):
                    #torch.cuda.empty_cache()
                    attribution = only_calc_attribution(img, instance, method, args.method)
                    total_attribution += attribution
                total_attribution = total_attribution / float(len(args.model))
                img_copy = img.cpu().numpy()[0]
                img_copy = np.transpose(img_copy, (1,2,0))
                fig = visualization.visualize_image_attr(attr=total_attribution, original_image=img_copy, method="blended_heat_map", use_pyplot=False,cmap="YlOrRd") # This flag as false will allow saving
                fig[0].savefig(save_path)
                # ACTUALLY CORRECT ONE
                if len(args.model) != 1:
                    raise NotImplementedError("Fix the fstring below to not just take the first model")
                save_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../.{args.method}/min_occ{args.min_occ}-{args.model[0]}-{idx}-actual-{instance}-{pred}.png"))
                total_attribution = np.zeros(img[0].permute(1,2,0).shape)
                for _ in range(len(args.model)):
                    #torch.cuda.empty_cache()
                    pred = net(img)
                    pred = int(pred.argmax())
                    attribution = only_calc_attribution(img, instance, method, args.method)
                    total_attribution += attribution
                total_attribution = total_attribution / float(len(args.model))
                img_copy = img.cpu().numpy()[0]
                img_copy = np.transpose(img_copy, (1,2,0))
                fig = visualization.visualize_image_attr(attr=total_attribution, original_image=img_copy, method="blended_heat_map", use_pyplot=False, cmap="YlOrRd") # This flag as false will allow saving
                fig[0].savefig(save_path)
    else:
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            instance = int(batch['instance'][0])
            img = batch['img']
            img = img.to(args.device)
            pred = net(img)
            pred = int(pred.argmax())
            if pred == instance:
                save_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../.{args.method}/min_occ{args.min_occ}-{args.model[0]}-{idx}-correct-{instance}.png"))
                visualise_method(img, instance, save_path, method, args.method)
            else:                   # If this prediction is wrong, save both the wrong prediction, and the correct one there should have been
                save_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../.{args.method}/min_occ{args.min_occ}-{args.model[0]}-{idx}-wrong-{instance}-{pred}.png"))
                visualise_method(img, pred, save_path, method, args.method)
                save_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../.{args.method}/min_occ{args.min_occ}-{args.model[0]}-{idx}-actual-{instance}-{pred}.png"))
                visualise_method(img, instance, save_path, method, args.method)
