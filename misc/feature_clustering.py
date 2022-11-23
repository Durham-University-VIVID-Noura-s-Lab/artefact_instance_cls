import os
import argparse
import json

from tqdm import tqdm

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.lines import Line2D

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["PCA","TSNE","UMAP"], type=str, required=True, help="PCA, TSNE, or UMAP")
args = parser.parse_args()

feats_path = os.path.abspath("../.feats")
all_feats_paths = []
all_feats_paths.append(os.path.join(feats_path, "enetb6_min_occ3.json"))
all_feats_paths.append(os.path.join(feats_path, "enetb3_min_occ4.json"))
all_feats_paths.append(os.path.join(feats_path, "enetb4_min_occ5.json"))
all_feats_paths.append(os.path.join(feats_path, "enetb6_min_occ6.json"))

all_feats = []
for feat_path in all_feats_paths:
    with open(feat_path, "r") as f:
        feats = json.load(f)
        all_feats.append(feats)

class_colours = {
    "oriental":"r",
    "egyptian":"g",
    "castle":"b",
    "fulling_mill":"Black"
}

legend_elements = [
    Line2D([0], [0], marker='o', color='r', label='Oriental', markerfacecolor='r'),
    Line2D([0], [0], marker='o', color='g', label='Egyptian', markerfacecolor='g'),
    Line2D([0], [0], marker='o', color='b', label='Castle', markerfacecolor='b'),
    Line2D([0], [0], marker='o', color='Black', label='Fulling Mill', markerfacecolor='Black')
]

label_switch = {
    "oriental":"Oriental",
    "egyptian":"Egyptian",
    "castle":"Castle",
    "fulling_mill":"Fulling Mill"   
}


# all_feats
new_feats = [[],[],[],[]]
classes = [[],[],[],[]]

for idx in range(len(new_feats)):
    for (cls, feat) in all_feats[idx]:
        cls = cls.split("|")[0]
        classes[idx].append(cls)
        new_feats[idx].append(feat)
    new_feats[idx] = np.stack(new_feats[idx])

fig, axs = plt.subplots(2,2)

for idx in range(len(new_feats)):
    ax = axs[idx//2,idx%2]
    min_occ = 3+idx
    if args.method == "PCA":
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(new_feats[idx])
        for pc_idx, pc in tqdm(enumerate(pcs), total=len(pcs)):
            cls = classes[idx][pc_idx]
            if cls != "@NULL_CLASS@":
                ax.plot(pc[0], pc[1], marker="o", markersize=2, color=class_colours[cls], label=label_switch[cls])
        ax.set_title(f"PCA of Model Features (Minimum Answer Occurrence = {min_occ})", fontweight="bold")
        ax.set_xlabel("Principal Component 1", fontweight="bold")
        ax.set_ylabel("Principal Component 2", fontweight="bold")
        #ax.set_xticks(fontweight="bold")
        #ax.set_yticks(fontweight="bold")
        #ax.setp(ax.spines.values(), linewidth=3)
        ax.xaxis.set_tick_params(width=3)
        ax.yaxis.set_tick_params(width=3)
        ax.grid()
    elif args.method == "TSNE":
        tsne = TSNE(n_components=2)
        tsnes = tsne.fit_transform(new_feats[idx])
        for pc_idx, x in enumerate(tsnes):
            cls = classes[idx][pc_idx]
            if cls != "@NULL_CLASS@":
                ax.plot(x[0], x[1], marker="o", markersize=2, color=class_colours[cls])
        ax.set_title(f"t-SNE of Model Features (Minimum Answer Occurrence = {min_occ})", fontweight="bold")
        ax.set_xlabel("Dim 1", fontweight="bold")
        ax.set_ylabel("Dim 2", fontweight="bold")
        #ax.xticks(fontweight="bold")
        #ax.yticks(fontweight="bold")
        #ax.setp(ax.spines.values(), linewidth=3)
        ax.xaxis.set_tick_params(width=3)
        ax.yaxis.set_tick_params(width=3)
        ax.grid()
    elif args.method == "UMAP":
        # UMAP
        mappy = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine')#metric='cosine')
        #NOTE NOTE NOTE UMAP can't deal with much higher than 4050 for some reason, throws a SegFault
        umaps = mappy.fit_transform(new_feats[idx])
        for pc_idx, x in enumerate(umaps):
            cls = classes[idx][pc_idx]
            if cls != "@NULL_CLASS@":
                ax.plot(x[0], x[1], marker="o", markersize=2, color=class_colours[cls])
        ax.set_title(f"UMAP of Model Features (Minimum Answer Occurrence = {min_occ})", fontweight="bold")
        ax.set_xlabel("Dim 1", fontweight="bold")
        ax.set_ylabel("Dim 2", fontweight="bold")
        #ax.xticks(fontweight="bold")
        #ax.yticks(fontweight="bold")
        #ax.setp(ax.spines.values(), linewidth=3)
        ax.xaxis.set_tick_params(width=3)
        ax.yaxis.set_tick_params(width=3)
        ax.grid()
    ax.legend(handles=legend_elements, loc="best")
plt.show()
