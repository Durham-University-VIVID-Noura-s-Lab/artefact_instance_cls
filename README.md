# Code Usage
1. ```git clone https://github.com/Jumperkables/archaeology```
2. Follow the sub-instructions to download ResNet rescaled
    * I cannot host this code directly on GitHub as i cannot confirm it is on the MiT License, and it would be a breach of copyright. I do not imply or recommend in any way that you host this publically available code yourself on any public platform.
    * Create the file ```models/resnet\_rescaled.py```
    * Put the following imports at the top of the file:
    ```
    import torch
    import timm
    import wandb
    import torchvision
    import pandas as pd
    import torch.nn as nn

    from torchvision import transforms
    from accelerate import Accelerator
    from tqdm import tqdm
    from timm.models.registry import register_model
    from timm.models.helpers import build_model_with_cfg
    from timm.models.resnet import Bottleneck, _create_resnet, default_cfgs, _cfg, make_blocks, create_classifier
    ```
    * From [this implementation](https://colab.research.google.com/drive/1RVOvZ7AkJuV8WNJwkXxxTtByEIXVV6CC?usp=sharing#scrollTo=bWioF_21jntW), copy the classes and functions ```ResNet```,```_create_resnet```, and all ```@register``` functions into ```models/resnet\_rescaled.py```
3. ```pip install -r requirements.txt```
4. ```mkdir data .results .feats```
5. Acquire the oriental museum dataset and put it in ```data```
    * Run the ```CLEAN_OM.py``` script.
    * Note that the exact dataset/filenames of the dataset that users aquire may have changed. We cannot host this dataset here, but may be able to verify if the splits received are similar to those in our experiments.
6. Experiments
    * All experiments can be ran via bash scripts in the ```scripts/``` folder
