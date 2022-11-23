import os
import sys
import argparse
import ipdb
from glob import glob
import copy
import random
import cv2
from tqdm import tqdm
import json

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import timm

import numpy as np
import pytorch_lightning as pl
import torchmetrics
import wandb
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Local imports
#import models.resnet_rescaled as resnet_rescaled
from datasets import GeneralDataset, GeneralDatasetPreloaded
import utils

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


class Ensemble(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, metadata_dicts: dict, models: list):
        super().__init__()
        self.args = args
        self.metadata_dicts = metadata_dicts
        assert args.metadata == ["instance"], f"Not working for non-instance classification"
        # NOTE "occurrences" end up in metadata_dicts when metadata == "instance". IT DOES NOT NEED SPECIFYING. IT IS INTERNALLY HANDLED ELSEWHERE
        if "occurrences" in metadata_dicts.keys():
            self.occurrences = self.metadata_dicts.pop("occurrences")   # self.occurrences holds the number of occurrences of each instance across all sets
        self.models = models
        for model in self.models:
            model.freeze()
        self.classifier = nn.Linear(10, len(models))

        # Metrics
        self.acc = torchmetrics.Accuracy()
        self.acc_top3 = torchmetrics.Accuracy(top_k=3)
        self.acc_top5 = torchmetrics.Accuracy(top_k=5)
        self.acc_top10 = torchmetrics.Accuracy(top_k=10)
        #self.f1 = torchmetrics.F1Score(num_classes=len(self.metadata_dicts['instance']), average="macro", mdmc_average="global")
        #self.recall = torchmetrics.Recall(num_classes=len(self.metadata_dicts['instance']), average="macro", mdmc_average="global")
        #self.prec = torchmetrics.Precision(num_classes=len(self.metadata_dicts['instance']), average="macro", mdmc_average="global")

        # Test predictions are saved to calculate accuracy per num of instances etc...
        self.test_pred = []
        
        # Resizer
        self.resize_switch = {
            "enetb0":transforms.Resize((256,224)),
            "enetb1":transforms.Resize((256,240)),
            "enetb2":transforms.Resize((288,288)),
            "enetb3":transforms.Resize((320,300)),
            "enetb4":transforms.Resize((384,380)),
            "enetb5":transforms.Resize((489,456)),
            "enetb6":transforms.Resize((561,528)),    # Original
            "enetb7":transforms.Resize((633,600)),    # Original
            "inceptionv3":transforms.Resize((299,299)),
            "inceptionv4":transforms.Resize((299,299)),
            "resnet_rs_101":transforms.Resize((224,224))
        }

        if self.args.conf_matrix:
            self.test_feats = []
            self.idx2instance = {v:k for k,v in self.metadata_dicts['instance'].items()}


    def forward(self, inputs):
        outputs = []
        for idx, model in enumerate(self.models):
            inputs = self.resize_switch[self.args.model[idx]](inputs)
            mod_output = model(inputs)
            outputs.append(mod_output)
        return outputs


    def configure_optimizers(self):
        lr = 10**(self.args.lr)
        optimizer_name = self.args.optimiser # ["Adam", "RMSprop", "SGD"])
        optimizer = getattr(torch.optim, optimizer_name)(self.parameters(), lr=lr)
        return optimizer


    def test_step_conf_matrix(self, test_batch, batch_idx):
        img = test_batch['img']
        enc_out = self(img)
        enc_out = enc_out[0]
        enc_out = enc_out.cpu()
        instance = int(test_batch['instance'].cpu())
        instance = self.idx2instance[instance]
        self.test_feats.append([instance, enc_out])


    def test_step(self, test_batch, batch_idx):
        if self.args.conf_matrix:
            self.test_step_conf_matrix(test_batch, batch_idx)
            return(0)
        img = test_batch['img']
        out = self.forward(img)
        label = test_batch['instance']
        # Softmax the outputs, sum and vote softmax again
        vote_out = torch.zeros(out[0].shape)
        vote_out = vote_out.to(self.device)
        for o in out:
            vote_out += torch.softmax(o, dim=1)
        vote_out = torch.softmax(vote_out, dim=1)
        self.log(f"test_instance_acc", self.acc(vote_out, label), prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"test_instance_acc_top3", self.acc_top3(vote_out, label), prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"test_instance_acc_top5", self.acc_top5(vote_out, label), prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"test_instance_acc_top10", self.acc_top10(vote_out, label), prog_bar=False, on_step=False, on_epoch=True)
        #self.log(f"test_instance_f1", self.f1(vote_out.argmax(dim=1), label), on_step=False, on_epoch=True)
        #self.log(f"test_instance_recall", self.recall(vote_out.argmax(dim=1), label), on_step=False, on_epoch=True)
        #self.log(f"test_instance_prec", self.prec(vote_out.argmax(dim=1), label), on_step=False, on_epoch=True)
        self.test_pred.append({"logits":vote_out.cpu(), "pred":torch.argmax(vote_out, dim=1).cpu(), "gt":test_batch['instance'].cpu()})


    def test_epoch_end(self, test_step_outputs):
        # This is triggered at the end of the testing step. Currently only does something for instance classification
        # Invert the instance2idx dictionary to get idx2instance
        idx2instance = {v:k for k,v in self.metadata_dicts['instance'].items()}
        if self.args.conf_matrix:
            n_classes = int(self.test_feats[0][1].shape[1])
            heatmap = torch.zeros(n_classes, n_classes)
            #ax = plt.axes()
            plt.title("Confusion Matrix\nTest Set Images-Per-Instance >= 6")
            plt.xlabel("Predicted Object")
            plt.ylabel("Actual Object")
            for pred in self.test_feats:
                label = pred[0]
                label = self.metadata_dicts['instance'][label]
                logits = pred[1][0]
                choice = int(logits.softmax(dim=0).argmax())
                heatmap[label][choice] += 1
            flipped_heatmap = heatmap.permute(1,0)
            tps = [int(heatmap[xx][xx]) for xx in range(n_classes)]
            fps = [int(heatmap[xx].sum()) for xx in range(n_classes)]
            fps = [int(fps[xx]-tps[xx]) for xx in range(n_classes)]
            fns = [int(flipped_heatmap[xx].sum()) for xx in range(n_classes)]
            fns = [int(fns[xx]-tps[xx]) for xx in range(n_classes)]
            precs = [tps[xx]/(tps[xx]+fps[xx]) if tps[xx]+fps[xx] != 0 else None for xx in range(n_classes)]
            recalls = [tps[xx]/(tps[xx]+fns[xx]) if tps[xx]+fns[xx] != 0 else None for xx in range(n_classes) ]
            f1s = [ tps[xx]/(tps[xx]+0.5*(fps[xx]+fns[xx])) if (tps[xx]+0.5*(fps[xx]+fns[xx])) != 0 else None for xx in range(n_classes) ]
            class_counts = heatmap.sum(dim=1)
            prec_weights = [class_counts[xx] for xx in range(n_classes) if precs[xx] != None]
            prec_weights = torch.tensor(prec_weights)/sum(prec_weights)
            precs = torch.tensor([xx for xx in precs if xx != None])

            recall_weights = [class_counts[xx] for xx in range(n_classes) if recalls[xx] != None]
            recall_weights = torch.tensor(recall_weights)/sum(recall_weights)
            recalls = torch.tensor([xx for xx in recalls if xx != None])

            f1_weights = [class_counts[xx] for xx in range(n_classes) if f1s[xx] != None]
            f1_weights = torch.tensor(f1_weights)/sum(f1_weights)
            f1s = torch.tensor([xx for xx in f1s if xx != None])

            prec = torch.dot(precs, prec_weights)
            recall = torch.dot(recalls, recall_weights)
            f1 = torch.dot(f1s, f1_weights)
            self.log(f"test_prec", prec)
            self.log(f"test_recall", recall)
            self.log(f"test_f1", f1)
            #return 0
        # Count the number of examples for each occurrence threshold
        occ_3 = 0
        occ_4 = 0
        occ_5 = 0
        occ_6 = 0
        occ_7 = 0
        occ_8 = 0
        occ_9 = 0
        occ_10plus = 0  # occurrences of 10 or greater
        for pred_dict in tqdm(self.test_pred, total=len(self.test_pred)):
            idx = int(pred_dict['gt'])
            # TODO Update after torch support python3.10 and i can use switch statements
            if idx2instance[idx] != "@NULL_CLASS@": # "@NULL_CLASS@" is a unique identifier used in dataset.py to have to hold the total number
                occ = self.occurrences[idx2instance[idx]]
                if occ == 3:
                    occ_3 += 1
                    occ_str = "==3"
                elif occ == 4:
                    occ_4 += 1
                    occ_str = "==4"
                elif occ == 5:
                    occ_5 += 1
                    occ_str = "==5"
                elif occ == 6:
                    occ_6 += 1
                    occ_str = "==6"
                elif occ == 7:
                    occ_7 += 1
                    occ_str = "==7"
                elif occ == 8:
                    occ_8 += 1
                    occ_str = "==8"
                elif occ == 9:
                    occ_9 += 1
                    occ_str = "==9"
                elif occ >= 10:
                    occ_10plus += 1
                    occ_str = ">=10"
                else:
                    raise ValueError("Occurences should not have gotten below 3")
                # Log the test_acc
                self.log(f"test_occ{occ_str}_acc", self.acc(pred_dict['logits'], pred_dict['gt']))
                self.log(f"test_occ{occ_str}_acc_top3", self.acc_top3(pred_dict['logits'], pred_dict['gt']))
                self.log(f"test_occ{occ_str}_acc_top5", self.acc_top5(pred_dict['logits'], pred_dict['gt']))
                self.log(f"test_occ{occ_str}_acc_top10", self.acc_top10(pred_dict['logits'], pred_dict['gt']))

        biggus = max(set(self.metadata_dicts['instance'].values()))
        unseen = sum( [ int(sum(j['pred'] == biggus)) for j in self.test_pred] ) # Find the number of instances in the test set that don't have an instance appear in the train set
        self.log("test_unseen_count", unseen) 
        self.test_pred = []


class LMSystem(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, metadata_dicts: dict):
        super().__init__()
        fc_intermediate = args.fc_intermediate
        dropout = args.dropout
        bool_freeze = args.encoder_freeze
        self.args = args
        self.metadata_dicts = metadata_dicts
        # NOTE "occurrences" end up in metadata_dicts when metadata == "instance". IT DOES NOT NEED SPECIFYING. IT IS INTERNALLY HANDLED ELSEWHERE
        if "occurrences" in metadata_dicts.keys():
            self.occurrences = self.metadata_dicts.pop("occurrences")   # self.occurrences holds the number of occurrences of each instance across all sets
        # Model
        cnn_switch = {  # Pick the correct model
            "vgg16"     : torchvision.models.vgg16,
            "vgg19"     : torchvision.models.vgg19,
            "resnet18"  : torchvision.models.resnet18,
            "resnet34"  : torchvision.models.resnet34,
            "resnet50"  : torchvision.models.resnet50,
            "resnet101" : torchvision.models.resnet101,
            "resnet152" : torchvision.models.resnet152,
            "densenet161": torchvision.models.densenet161,
            "enetb0": torchvision.models.efficientnet_b0,
            "enetb1": torchvision.models.efficientnet_b1,
            "enetb2": torchvision.models.efficientnet_b2,
            "enetb3": torchvision.models.efficientnet_b3,
            "enetb4": torchvision.models.efficientnet_b4,
            "enetb5": torchvision.models.efficientnet_b5,
            "enetb6": torchvision.models.efficientnet_b6,
            "enetb7": torchvision.models.efficientnet_b7,
            #"resnet_rs_101": resnet_rescaled.resnetrs101,
            #"resnet_rs_152": resnet_rescaled.resnetrs152,
            "inceptionv3": torchvision.models.inception_v3,
            "inceptionv4":None,
        }
        fc_switch = {   # Figure out the needed size of FC layer from the CNN feature extraction layer
            "vgg16"     : 25088,
            "vgg19"     : 25088,
            "resnet18"  : 512,
            "resnet34"  : 512,
            "resnet50"  : 2048,
            "resnet101" : 2048,
            "resnet152" : 2048,
            "densenet161": 2208,
            "enetb0":1280,
            "enetb1":1280,
            "enetb2":1408,
            "enetb3":1536,
            "enetb4":1792,
            "enetb5":2048,
            "enetb6":2304,
            "enetb7":2560,
            "san":2048,
            "inceptionv3":2048,
            "inceptionv4":1536,
        }
        # TODO REMOVE THE FINAL CNN layer
        fc_out = fc_switch["".join(args.model.split("_rs_"))]
        # This layer represents predictions for each piece of metadata
        self.metadata_fc = {}
        for key, md in self.metadata_dicts.items():
            self.metadata_fc[key] = len(set(md.values()))
        # Fully connected sequence
        self.fc = nn.Sequential(
            nn.BatchNorm1d(fc_out),
            nn.Linear(fc_out, fc_intermediate),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_intermediate, sum(self.metadata_fc.values())), # Final classification layer is the size of the sum of all classes from all metadata
        )
        print("\n\nFC INTERMEDIATE",sum(self.metadata_fc.values()),"\n\n")

        # Remove the fully connected layer final layer of the CNN
        if "_rs_" in args.model:
            self.encoder = timm.create_model(args.model.replace("_",""), pretrained=True)
        elif self.args.model == "inceptionv4":
            self.encoder = timm.create_model("inception_v4", pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))
        elif self.args.model == "san":
            #NOTE THEIR DEFAULT VALUES self.encoder = san(sa_type=0, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7], num_classes=sum(self.metadata_fc.values()))
            self.encoder = san(sa_type=0, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7], num_classes=sum(self.metadata_fc.values()))
        elif self.args.model.startswith("inceptionv"):
            # NOTE WE DISABLE THE AUX LOGITS
            self.encoder = torch.nn.Sequential(*(list(cnn_switch[args.model](pretrained=True, aux_logits=False).children())[:-1]))
        else:
            self.encoder = torch.nn.Sequential(*(list(cnn_switch[args.model](pretrained=True).children())[:-1]))

        # Freeze the encoder
        if bool_freeze:
            print("\n\nFREEZING ENCODER!\n\n")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print("\n\nENCODER IS TRAINABLE!\n\n")

        # Criterion and metrics
        if self.args.loss_weight_scaling:
            # args.loss_weight_scaling forces weight scaling on classes
            assert self.args.metadata == ["instance"], f"Loss scaling currently not supported for non-instance data"
            # TODO Clean up these if-else clauses, they are unnecessary
            if "@NULL_CLASS@" in self.metadata_dicts['instance'].keys():
                final_index = self.metadata_dicts['instance']["@NULL_CLASS@"]
            else:
                final_index = max(self.metadata_dicts['instance'].values())
            instances_weighting = torch.zeros(final_index+1)    # +1 for indexing offset
            total = sum(self.occurrences.values())  # Total number of images
            for instance, occ in self.occurrences.items():  # loss_weight_scaling calculated here
                i_idx = self.metadata_dicts['instance'][instance]
                weight = total / occ
                instances_weighting[i_idx] = weight
            instances_weighting = instances_weighting/instances_weighting.max()
            if "@NULL_CLASS@" in self.metadata_dicts['instance']:
                final_index = self.metadata_dicts['instance']['@NULL_CLASS@']
                self.criterion = nn.NLLLoss(ignore_index=final_index, weight=instances_weighting) # Ignore the index of the unseen class
            else:
                self.criterion = nn.NLLLoss(weight=instances_weighting)
        else:
            print("\n\nThe NLLLoss object is UNSCALED CURRENTLY\n\n")
            self.criterion = nn.NLLLoss()   # TODO Ignore the index of the unseen class here too

        # Metrics
        self.acc = torchmetrics.Accuracy()
        self.acc_top3 = torchmetrics.Accuracy(top_k=3)
        self.acc_top5 = torchmetrics.Accuracy(top_k=5)
        self.acc_top10 = torchmetrics.Accuracy(top_k=10)
        #self.f1 = torchmetrics.F1(num_classes=len(self.metadata_dicts['instance']), average="macro", mdmc_average="global")
        #self.recall = torchmetrics.Recall(num_classes=len(self.metadata_dicts['instance']), average="macro", mdmc_average="global")
        #self.prec = torchmetrics.Precision(num_classes=len(self.metadata_dicts['instance']), average="macro", mdmc_average="global")

        # Test predictions are saved to calculate accuracy per num of instances etc...
        self.test_pred = []

        # Save the features generated from the test set
        assert not(self.args.feats_gen == self.args.conf_matrix == True), "You cannot do both feats_gen and conf_matrix at the same time"
        if self.args.conf_matrix:
            self.test_feats = []
            self.idx2instance = {v:k for k,v in self.metadata_dicts['instance'].items()}
        if self.args.feats_gen:
            self.test_feats = []
            self.idx2instance = {v:k for k,v in self.metadata_dicts['instance'].items()}
            self.save_feats_gen_path = f".feats/{self.args.model}_min_occ{self.args.min_occ}.json"
            if os.path.exists(self.save_feats_gen_path):
                raise Exception("Gen feats file already found:", self.save_feats_gen_path)

    def forward(self, inputs):
        out = self.encoder(inputs)
        if self.args.model != "san":    # All models that aren't san require flattening and an FC unit
            enc_out = torch.flatten(out, start_dim=1)
            out = self.fc(enc_out)
        if self.args.feats_gen:
            return enc_out
        else:
            return out

    def configure_optimizers(self):
        lr = 10**(self.args.lr)
        optimizer_name = self.args.optimiser # ["Adam", "RMSprop", "SGD"])
        optimizer = getattr(torch.optim, optimizer_name)(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img = train_batch['img']
        out = self(img)
        # This is to handle multiple losses IF there are multiple metadata specified
        start = 0
        train_losses = []
        for key, sze in self.metadata_fc.items():
            label = train_batch[key]
            subout = out[:,start:start+sze]
            subout = F.log_softmax(subout, dim=1)
            loss = self.criterion(subout, train_batch[key])
            train_losses.append(loss)
            self.log(f"train_{key}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"train_{key}_acc", self.acc(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"train_{key}_acc_top3", self.acc_top3(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"train_{key}_acc_top5", self.acc_top5(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"train_{key}_acc_top10", self.acc_top10(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            #self.log(f"train_{key}_f1", self.f1(subout.argmax(dim=1), label), on_step=False, on_epoch=True)
            #self.log(f"train_{key}_recall", self.recall(subout.argmax(dim=1), label), on_step=False, on_epoch=True)
            #self.log(f"train_{key}_prec", self.prec(subout.argmax(dim=1), label), on_step=False, on_epoch=True)
            start += sze
        return sum(train_losses)

    def validation_step(self, valid_batch, batch_idx):
        img = valid_batch['img']
        out = self(img)
        # Handling multiple losses IF specified in args.metadata
        start = 0
        valid_losses = []
        for key, sze in self.metadata_fc.items():
            label = valid_batch[key]
            subout = out[:,start:start+sze]
            subout = F.log_softmax(subout, dim=1)
            loss = self.criterion(subout, valid_batch[key])
            valid_losses.append(loss)
            self.log(f"valid_{key}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"valid_{key}_acc", self.acc(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"valid_{key}_acc_top3", self.acc_top3(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"valid_{key}_acc_top5", self.acc_top5(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"valid_{key}_acc_top10", self.acc_top10(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            #self.log(f"valid_{key}_f1", self.f1(subout.argmax(dim=1), label), on_step=False, on_epoch=True)
            #self.log(f"valid_{key}_recall", self.recall(subout.argmax(dim=1), label), on_step=False, on_epoch=True)
            #self.log(f"valid_{key}_prec", self.prec(subout.argmax(dim=1), label), on_step=False, on_epoch=True)
            start += sze
        return sum(valid_losses)

    def test_step(self, test_batch, batch_idx):
        if self.args.feats_gen:
            self.test_step_feats_gen(test_batch, batch_idx)
            return(0)
        elif self.args.conf_matrix:
            self.test_step_conf_matrix(test_batch, batch_idx)
            return(0)
        img = test_batch['img']
        out = self(img)
        start = 0
        test_losses = []
        for key, sze in self.metadata_fc.items():
            label = test_batch[key]
            subout = out[:,start:start+sze]
            subout = F.log_softmax(subout, dim=1)
            loss = self.criterion(subout, test_batch[key])
            test_losses.append(loss)
            self.log(f"test_{key}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"test_{key}_acc", self.acc(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"test_{key}_acc_top3", self.acc_top3(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"test_{key}_acc_top5", self.acc_top5(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"test_{key}_acc_top10", self.acc_top10(subout, label), prog_bar=False, on_step=False, on_epoch=True)
            #self.log(f"test_{key}_f1", self.f1(subout.argmax(dim=1), label), on_step=False, on_epoch=True)
            #self.log(f"test_{key}_recall", self.recall(subout.argmax(dim=1), label), on_step=False, on_epoch=True)
            #self.log(f"test_{key}_prec", self.prec(subout.argmax(dim=1), label), on_step=False, on_epoch=True)
            self.test_pred.append({"logits":subout.cpu(), "pred":torch.argmax(subout, dim=1).cpu(), "gt":test_batch[key].cpu()})
            start += sze
        return sum(test_losses)

    def test_step_feats_gen(self, test_batch, batch_idx):
        img = test_batch['img']
        enc_out = self(img)
        enc_out = enc_out[0]
        enc_out = enc_out.cpu().tolist()
        instance = int(test_batch['instance'].cpu())
        instance = self.idx2instance[instance]
        self.test_feats.append([instance, enc_out])

    def test_step_conf_matrix(self, test_batch, batch_idx):
        img = test_batch['img']
        enc_out = self(img)
        enc_out = enc_out[0]
        enc_out = enc_out.cpu()
        instance = int(test_batch['instance'].cpu())
        instance = self.idx2instance[instance]
        self.test_feats.append([instance, enc_out])


    def test_recall_prec_f1(self, subsets=["oriental", "egyptian", "fulling_mill", "castle"]):
        n_classes = int(self.test_feats[0][1].shape[0])
        heatmap = torch.zeros(n_classes, n_classes)
        for pred in self.test_feats:
            label = pred[0]
            if label.split("|")[0] in subsets:
                label = self.metadata_dicts['instance'][label]
                logits = pred[1]
                choice = int(logits.softmax(dim=0).argmax())
                heatmap[label][choice] += 1
        flipped_heatmap = heatmap.permute(1,0)
        tps = [int(heatmap[xx][xx]) for xx in range(n_classes)]
        fps = [int(heatmap[xx].sum()) for xx in range(n_classes)]
        fps = [int(fps[xx]-tps[xx]) for xx in range(n_classes)]
        fns = [int(flipped_heatmap[xx].sum()) for xx in range(n_classes)]
        fns = [int(fns[xx]-tps[xx]) for xx in range(n_classes)]
        precs = [tps[xx]/(tps[xx]+fps[xx]) if tps[xx]+fps[xx] != 0 else None for xx in range(n_classes)]
        recalls = [tps[xx]/(tps[xx]+fns[xx]) if tps[xx]+fns[xx] != 0 else None for xx in range(n_classes) ]
        f1s = [ tps[xx]/(tps[xx]+0.5*(fps[xx]+fns[xx])) if (tps[xx]+0.5*(fps[xx]+fns[xx])) != 0 else None for xx in range(n_classes) ]
        class_counts = heatmap.sum(dim=1)
        prec_weights = [class_counts[xx] for xx in range(n_classes) if precs[xx] != None]
        prec_weights = torch.tensor(prec_weights)/sum(prec_weights)
        precs = torch.tensor([xx for xx in precs if xx != None])

        recall_weights = [class_counts[xx] for xx in range(n_classes) if recalls[xx] != None]
        recall_weights = torch.tensor(recall_weights)/sum(recall_weights)
        recalls = torch.tensor([xx for xx in recalls if xx != None])

        f1_weights = [class_counts[xx] for xx in range(n_classes) if f1s[xx] != None]
        f1_weights = torch.tensor(f1_weights)/sum(f1_weights)
        f1s = torch.tensor([xx for xx in f1s if xx != None])

        prec = torch.dot(precs, prec_weights)
        recall = torch.dot(recalls, recall_weights)
        f1 = torch.dot(f1s, f1_weights)
        self.log(f"test_{subsets}_prec", prec)
        self.log(f"test_{subsets}_recall", recall)
        self.log(f"test_{subsets}_f1", f1)


    def test_epoch_end(self, test_step_outputs):
        if self.args.feats_gen:
            with open(self.save_feats_gen_path, "w") as f:
                json.dump(self.test_feats, f)
            return None

        if self.args.conf_matrix:
            # TODO Remove this section of code and replace with the above method which is identical
            self.test_recall_prec_f1(["oriental"])
            self.test_recall_prec_f1(["egyptian"])
            self.test_recall_prec_f1(["castle"])
            self.test_recall_prec_f1(["fulling_mill"])
            n_classes = int(self.test_feats[0][1].shape[0])
            heatmap = torch.zeros(n_classes, n_classes)
            #ax = plt.axes()
            #plt.title("Confusion Matrix\nTest Set Images-Per-Instance >= 6")
            #plt.xlabel("Predicted Object")
            #plt.ylabel("Actual Object")
            for pred in self.test_feats:
                label = pred[0]
                label = self.metadata_dicts['instance'][label]
                logits = pred[1]
                choice = int(logits.softmax(dim=0).argmax())
                heatmap[label][choice] += 1
            flipped_heatmap = heatmap.permute(1,0)
            tps = [int(heatmap[xx][xx]) for xx in range(n_classes)]
            fps = [int(heatmap[xx].sum()) for xx in range(n_classes)]
            fps = [int(fps[xx]-tps[xx]) for xx in range(n_classes)]
            fns = [int(flipped_heatmap[xx].sum()) for xx in range(n_classes)]
            fns = [int(fns[xx]-tps[xx]) for xx in range(n_classes)]
            precs = [tps[xx]/(tps[xx]+fps[xx]) if tps[xx]+fps[xx] != 0 else None for xx in range(n_classes)]
            recalls = [tps[xx]/(tps[xx]+fns[xx]) if tps[xx]+fns[xx] != 0 else None for xx in range(n_classes) ]
            f1s = [ tps[xx]/(tps[xx]+0.5*(fps[xx]+fns[xx])) if (tps[xx]+0.5*(fps[xx]+fns[xx])) != 0 else None for xx in range(n_classes) ]
            class_counts = heatmap.sum(dim=1)
            prec_weights = [class_counts[xx] for xx in range(n_classes) if precs[xx] != None]
            prec_weights = torch.tensor(prec_weights)/sum(prec_weights)
            precs = torch.tensor([xx for xx in precs if xx != None])

            recall_weights = [class_counts[xx] for xx in range(n_classes) if recalls[xx] != None]
            recall_weights = torch.tensor(recall_weights)/sum(recall_weights)
            recalls = torch.tensor([xx for xx in recalls if xx != None])

            f1_weights = [class_counts[xx] for xx in range(n_classes) if f1s[xx] != None]
            f1_weights = torch.tensor(f1_weights)/sum(f1_weights)
            f1s = torch.tensor([xx for xx in f1s if xx != None])

            prec = torch.dot(precs, prec_weights)
            recall = torch.dot(recalls, recall_weights)
            f1 = torch.dot(f1s, f1_weights)
            self.log(f"test_prec", prec)
            self.log(f"test_recall", recall)
            self.log(f"test_f1", f1)
            #plt.imshow(heatmap, "gist_gray", vmin=0, vmax=1)#cmap="Greys", interpolation="nearest")
            #plt.show()
            #return(0)
            
        # This is triggered at the end of the testing step. Currently only does something for instance classification
        if args.metadata == ["instance"]:
            # Invert the instance2idx dictionary to get idx2instance
            idx2instance = {v:k for k,v in self.metadata_dicts['instance'].items()}
            # Count the number of examples for each occurrence threshold
            occ_3 = 0
            occ_4 = 0
            occ_5 = 0
            occ_6 = 0
            occ_7 = 0
            occ_8 = 0
            occ_9 = 0
            occ_10plus = 0  # occurrences of 10 or greater
            n_oriental = 0
            n_egyptian = 0
            n_castle = 0
            n_fulling_mill = 0
            for pred_dict in tqdm(self.test_pred, total=len(self.test_pred)):
                idx = int(pred_dict['gt'])
                # TODO Update after torch support python3.10 and i can use switch statements
                if idx2instance[idx] != "@NULL_CLASS@": # "@NULL_CLASS@" is a unique identifier used in dataset.py to have to hold the total number
                    occ = self.occurrences[idx2instance[idx]]
                    if occ == 3:
                        occ_3 += 1
                        occ_str = "==3"
                    elif occ == 4:
                        occ_4 += 1
                        occ_str = "==4"
                    elif occ == 5:
                        occ_5 += 1
                        occ_str = "==5"
                    elif occ == 6:
                        occ_6 += 1
                        occ_str = "==6"
                    elif occ == 7:
                        occ_7 += 1
                        occ_str = "==7"
                    elif occ == 8:
                        occ_8 += 1
                        occ_str = "==8"
                    elif occ == 9:
                        occ_9 += 1
                        occ_str = "==9"
                    elif occ >= 10:
                        occ_10plus += 1
                        occ_str = ">=10"
                    else:
                        raise ValueError("Occurences should not have gotten below 3")
                    instance_cls = idx2instance[idx].split("|")[0]
                    if instance_cls == "oriental":
                        n_oriental += 1
                    elif instance_cls == "egyptian":
                        n_egyptian += 1
                    elif instance_cls == "castle":
                        n_castle += 1
                    elif instance_cls == "fulling_mill":
                        n_fulling_mill += 1
                    else:
                        raise ValueError("Instance should have belonged to one of the above 4 classes (in code)")
                    # Log the test_acc
                    self.log(f"test_occ{occ_str}_acc", self.acc(pred_dict['logits'],pred_dict['gt']))
                    self.log(f"test_occ{occ_str}_acc_top3", self.acc_top3(pred_dict['logits'],pred_dict['gt']))
                    self.log(f"test_occ{occ_str}_acc_top5", self.acc_top5(pred_dict['logits'],pred_dict['gt']))
                    self.log(f"test_occ{occ_str}_acc_top10", self.acc_top10(pred_dict['logits'],pred_dict['gt']))
                    # and by class
                    self.log(f"test_occ{instance_cls}_acc", self.acc(pred_dict['logits'],pred_dict['gt']))
                    self.log(f"test_occ{instance_cls}_acc_top3", self.acc_top3(pred_dict['logits'],pred_dict['gt']))
                    self.log(f"test_occ{instance_cls}_acc_top5", self.acc_top5(pred_dict['logits'],pred_dict['gt']))
                    self.log(f"test_occ{instance_cls}_acc_top10", self.acc_top10(pred_dict['logits'],pred_dict['gt']))

            biggus = max(set(self.metadata_dicts['instance'].values()))
            unseen = sum( [ int(sum(j['pred'] == biggus)) for j in self.test_pred] ) # Find the number of instances in the test set that don't have an instance appear in the train set
            self.log("test_unseen_count", unseen) 
            self.test_pred = []


#def objective(trial, args, metadata_dicts, wandb_logger, gpus, train_loader, valid_loader):
#    pl_system = LMSystem(args, metadata_dicts, trial)
#    #checkpoint_callback = pl.callbacks.ModelCheckpoint(
#    #    os.path.join(os.getcwd(), ".results", f"{args.dataset}-{args.metadata}{args.model}_trial_{trial.number}"), monitor="valid_instance_acc"
#    #)
#    trainer = pl.Trainer(logger=wandb_logger, gpus=gpus, max_epochs=100, callbacks=[PyTorchLightningPruningCallback(trial, monitor="valid_instance_accuracy")])
#    trainer.fit(pl_system, train_loader, valid_loader)
#    return wandb_logger.metrics[-1]["valid_instance_acc"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", choices=["vgg16","vgg19","resnet18","resnet34","resnet50","resnet101","resnet152","densenet161","resnet_rs_101", "san", "enetb0","enetb1","enetb2","enetb3","enetb4","enetb5","enetb6","enetb7","inceptionv3","inceptionv4"], required=True, type=str, help="model to use")
    parser.add_argument("--num_workers", default=0, type=int, help="number of pytorch workers for dataloader")
    parser.add_argument("--device", default=0, type=int, help="Cuda device, -1 for CPU")
    parser.add_argument("--optimiser", default="Adam", type=str, choices=["Adam", "RMSprop", "SGD"], help="Optimiser kind to use")
    parser.add_argument("--lr", default=-5, type=float, help="Exponent of the learning rate")
    parser.add_argument("--bsz", default=32, type=int, help="Training batch size")
    parser.add_argument("--vt_bsz", default=32, type=int, help="Validation and test batch size")
    parser.add_argument("--epochs", default=1, type=int, help="Epochs to run for")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle the dataset")
    parser.add_argument("--wandb", action="store_true", help="To enable online logging for wandb")
    parser.add_argument("--preload", action="store_true", help="Preload dataset. Run this when dataset loading is a bottleneck")
    parser.add_argument("--encoder_freeze", type=int, default=0, help="Freeze the CNN, SAN or other substantial part of the network, i.e. things that aren't the fully connected classifier. 0=dont freeze, 1=freeze")
    parser.add_argument("--dropout", type=float, default=0.2, help="FC dropout")
    parser.add_argument("--fc_intermediate", type=int, default=512, help="Intermediate size of the FC layers")
    parser.add_argument("--dataset", type=str, choices=["oriental-museum"], default="oriental_museum", help="""
    args.dataset
    """)
    parser.add_argument("--transforms", type=str, choices=["augment", "resize", "resize-by-model", "resize-no-normalisation", "none"], default="none", help="""
    The kind of transforms to use in preprocessing for images
    args.transforms
        resize = Crop down with normalisation and resize to 224
        resize-by-model = Resize the image basedon the one the pretrained model is used to
        resize-no-normalisation = Only crop down and resize
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
    parser.add_argument("--ensemble", action="store_true", help="Run a test-only ensemble of models")
    parser.add_argument("--min_occ", type=int, default=3, help="Minimum number of occurences required to keep the class for instances")
    parser.add_argument("--test_ckpt_path", nargs="+", type=str, default="", help="If specified, load the ckpt in and run it through the test set")
    parser.add_argument("--loss_weight_scaling", action="store_true", help="Activate this flag to enable weighted loss scaling for the multiclass problems")
    parser.add_argument("--feats_gen", action="store_true", help="If activated, store the final feature vectors generated before the linear layer (for visualisation purposes)")
    parser.add_argument("--conf_matrix", action="store_true", help="If activated, store the final feature vectors generated before the linear layer (for visualisation purposes)")
    args = parser.parse_args()

    # Initialise wandb logger, see args.wandb
    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"
    
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
    runname = f"btest-bsz{args.bsz}-lws{int(args.loss_weight_scaling)}-{args.transforms}-{args.dataset}-{args.metadata}"+min_occ_str+f"-seed{args.dset_seed}_{args.model}"
    wandb.init(entity="jumperkables", project="archaeology", name=runname, config=args)
    config = wandb.config
    wandb_logger = pl.loggers.WandbLogger()
    wandb_logger.log_hyperparams(config)#(args)

    if args.metadata == None:   # args.metadata holds a list of all metadata for classification
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
        data = dataset_switch[args.preload](args.dataset, args.shuffle, args.metadata, transforms=args.transforms, min_occ=args.min_occ, modelname=args.model)
        valid_data = copy.deepcopy(data)
        valid_data.choose_split("valid")
        test_data = copy.deepcopy(data)
        test_data.choose_split("test")
        train_data = data
        train_data.choose_split("train")
        metadata_dicts = train_data.metadata_dicts
    # Dataloader
    train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.bsz)
    valid_loader = DataLoader(valid_data, num_workers=args.num_workers, batch_size=args.vt_bsz)
    # NOTE for debugging test things faster # test_data = torch.utils.data.Subset(test_data, [i for i in range(10)])
    test_loader = DataLoader(test_data, num_workers=4, batch_size=1)

    # GPU
    if args.device == -1:
        gpus = None 
    else: 
        print("Add support for multiple GPUs")
        gpus = [args.device]

    # Ensembling
    if args.ensemble:
        models = []
        for i in range(len(args.model)):
            modeltype = args.model[i]
            model_checkpoint = args.test_ckpt_path[i]
            # Create a copy of args with the correct model in
            cargs = copy.deepcopy(args)
            cargs.model = modeltype
            model = LMSystem.load_from_checkpoint(model_checkpoint, args=cargs, metadata_dicts=copy.deepcopy(metadata_dicts))
            model.to(args.device)   # TODO multiple gpu support
            models.append(model)
        pl_system = Ensemble(args, metadata_dicts, models)
        trainer = pl.Trainer(logger=wandb_logger, gpus=gpus)
        trainer.test(model=pl_system, test_dataloaders=test_loader)
    else:   # Normal running
        # Checkpoint callbacks
        if len(args.metadata) > 1:
            # Where usually we classify multiple metadata at the same time, i have not continued support for it recently, and am thus raising this error
            raise NotImplementedError("Fix the callbacks to pick what to monitor in the case of multiple metadata classification")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=f"valid_{args.metadata[0]}_acc",
            dirpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), ".results"),  # Files saved in .results
            filename=f"{runname}"+'-{epoch:02d}',
            save_top_k=1,
            mode="max"
        )
        callbacks = [checkpoint_callback]
        # Run the pytorch lightning system
        # If checkpointing
        if args.test_ckpt_path != "":   # If you want to run ONLY TESTING on a pretrained model, then specifyi it's location with test_ckpt_path
            pl_system = LMSystem.load_from_checkpoint(args.test_ckpt_path, args=args, metadata_dicts=metadata_dicts)
            #ckpt = torch.load(args.test_ckpt_path)
            trainer = pl.Trainer(logger=wandb_logger, gpus=gpus)
            trainer.test(model=pl_system, test_dataloaders=test_loader)
        else:   # If test_ckpt_path isn't specified then it defaults to "", so run full training
            pl_system = LMSystem(args, metadata_dicts)
            trainer = pl.Trainer(logger=wandb_logger, gpus=gpus, max_epochs=args.epochs, callbacks=callbacks)
            trainer.fit(pl_system, train_loader, valid_loader)
            trainer.test(model=pl_system, test_dataloaders=test_loader)
