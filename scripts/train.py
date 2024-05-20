# Copyright 2022 Ryan Peters
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim 
import torch.nn as nn
import torch

from tqdm.auto import trange
import matplotlib.pyplot as plt
import argparse
import shutil
import time
import os

from torchvf.dataloaders import BPCIS, PhaseDataset, RgbDataset
from torchvf.dataloaders import MultiEpochsDataLoader
from torchvf.losses import IVPLoss, TverskyLoss
from torchvf.transforms import transforms
from torchvf.models import get_model
from torchvf.utils import *

from ml_collections.config_flags.config_flags import _ConfigFileParser

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
args = parser.parse_args()

FileParser = _ConfigFileParser(name="train")
cfg = FileParser.parse(args.config)

# When using GPU, have this on.
torch.backends.cudnn.benchmark = True

config_dir = os.path.join("./configs", cfg.CONFIG_PATH)
output_dir = next_model(cfg.WEIGHT_DIR)

shutil.copyfile(config_dir, os.path.join(output_dir, "config.py"))

######################################## 
############# DATA #####################
## make datasets for given datatype   ##
## and construct dataloader for them  ##
######################################## 
if cfg.DATA.TYPE == 'bpcis':
    train_dataset = BPCIS(
        cfg.DATA.DIR, 
        split = "train",
        vf = cfg.DATA.VF, 
        vf_delimiter = cfg.DATA.VF_DELIM,
        transforms = transforms[cfg.DATA.TRANSFORMS], 
        remove = cfg.DATA.REMOVE, 
        copy = cfg.DATA.COPY
    )
    valid_dataset = BPCIS(
        cfg.DATA.DIR, 
        split = "test",
        vf = cfg.DATA.VF, 
        vf_delimiter = cfg.DATA.VF_DELIM,
        transforms = transforms[cfg.DATA.TRANSFORMS], 
        remove = cfg.DATA.REMOVE, 
        copy = cfg.DATA.COPY
    )
elif cfg.DATA.TYPE == 'phase':
    all_dataset = PhaseDataset(
        cfg.DATA.DIR,
        vf=cfg.DATA.VF,
        vf_name=cfg.DATA.VF_DELIM,
        transforms=transforms[cfg.DATA.TRANSFORMS],
        remove=cfg.DATA.REMOVE,
        copy=cfg.DATA.COPY,
        expand=cfg.DATA.EXPAND,
        set_type=cfg.DATA.SET_TYPE
    )
    train_dataset, valid_dataset = torch.utils.data.random_split(
        all_dataset, [0.8, 0.2],
        generator=torch.Generator().manual_seed(cfg.DATA.RNG_SEED)
    )
elif cfg.DATA.TYPE == 'rgb':
    train_dataset = RgbDataset(
        os.path.join(cfg.DATA.DIR, 'train'),
        vf=cfg.DATA.VF,
        vf_delimiter=cfg.DATA.VF_DELIM,
        transforms=transforms[cfg.DATA.TRANSFORMS],
        remove=cfg.DATA.REMOVE,
        copy=cfg.DATA.COPY,
        expand=cfg.DATA.EXPAND
    )
    valid_dataset = RgbDataset(
        os.path.join(cfg.DATA.DIR, 'valid'),
        vf=cfg.DATA.VF,
        vf_delimiter=cfg.DATA.VF_DELIM,
        transforms=transforms[cfg.DATA.TRANSFORMS],
        remove=cfg.DATA.REMOVE,
        copy=cfg.DATA.COPY,
        expand=cfg.DATA.EXPAND
    )
else:
    raise ValueError('invalid data type')

# construct dataloaders for the train/valid datasets
train_dataloader = MultiEpochsDataLoader(
    train_dataset, 
    batch_size=cfg.BATCH_SIZE,
    pin_memory=True,
    drop_last=True,
    shuffle=True,
    num_workers=1
)
valid_dataloader = MultiEpochsDataLoader(
    valid_dataset,
    batch_size=cfg.BATCH_SIZE,
    pin_memory=True,
    drop_last=False,
    shuffle=False,
    num_workers=1
)

######################################## 
############ MODEL & OPTIM #############
## make model & optimizer
######################################## 

model = get_model(
    cfg.MODEL_TYPE,
    in_channels = cfg.DATA.C,
    out_channels = [1, 2],
    device = cfg.DEVICE
)

optimizer = optim.Adam(model.parameters(), lr = cfg.LR)

if cfg.PRETRAINED:
    model, optimizer, epoch = load_checkpoint(
        cfg.PRETRAINED_DIR, model, optimizer
    )

######################################## 
############ LOSS FUNCTIONS ############ 
######################################## 
vf_losses, sem_losses  = [], []

# vector field loss terms are IVP & MSE
if cfg.LOSS.IVP.APPLY:
    vf_losses.append([
        "IVP", 
        IVPLoss(
            dx = cfg.LOSS.IVP.DX,
            n_steps = cfg.LOSS.IVP.STEPS,
            solver = cfg.LOSS.IVP.SOLVER,
            device = cfg.DEVICE 
        ) 
    ])
    make_loss_csv(os.path.join(output_dir, "loss"), "IVP")
if cfg.LOSS.MSE.APPLY:
    vf_losses.append([
        "MSE", 
        nn.MSELoss()
    ])
    make_loss_csv(os.path.join(output_dir, "loss"), "MSE")
# semantic loss terms are Tversky & BCE
if cfg.LOSS.TVERSKY.APPLY:
    sem_losses.append([
        "Tversky",
        TverskyLoss(
            alpha = cfg.LOSS.TVERSKY.ALPHA, 
            beta = cfg.LOSS.TVERSKY.BETA, 
            from_logits = cfg.LOSS.TVERSKY.FROM_LOGITS 
        )
    ])
    make_loss_csv(os.path.join(output_dir, "loss"), "Tversky")
if cfg.LOSS.BCE.APPLY:
    sem_losses.append([
        "BCE", 
        nn.BCELoss()
    ])
    make_loss_csv(os.path.join(output_dir, "loss"), "BCE")

######################################## 
############ TRAINING LOOP #############
######################################## 
DEVICE = cfg.DEVICE
pbar = trange(1, cfg.EPOCHS+1, desc="Training")
if cfg.EARLY_STOP:
    early_stop = EarlyStopper(cfg.EARLY_STOP_PATIENCE,
                              cfg.EARLY_STOP_MINDELTA)
for epoch in pbar:
    epoch_time = time.time()
    # construct dictionary for keeping track of loss values
    loss_dict = {loss_name : {'train' : 0, 'valid' : 0}
                     for loss_name, _ in vf_losses+sem_losses}
    for mode in ['train', 'valid']:
        if mode == 'train':
            dataloader = train_dataloader
            model.train(True)
            grad_on = True
        else:
            dataloader = valid_dataloader
            model.eval()
            grad_on = False
        with torch.set_grad_enabled(grad_on):
            for step, (image, vf, inst_mask) in enumerate(dataloader, 1):
                step_time = time.time()
                image     = image.to(DEVICE, non_blocking=True).float()
                vf        = vf.to(DEVICE, non_blocking=True).float()
                inst_mask = inst_mask.to(DEVICE, non_blocking=True)
                semantic = torch.where(inst_mask > 0, 1.0, 0.0)
                # Reduces the number of memory operations. 
                for param in model.parameters():
                    param.grad = None
                # do the prediction
                pred_semantic, pred_vf = model(image)
                # calculate loss
                loss, loss_values = 0, []
                for name, loss_f in vf_losses:
                    loss_value = loss_f(pred_vf, vf)
                    loss += loss_value
                    loss_dict[name][mode] += loss_value.item()
                for name, loss_f in sem_losses:
                    loss_value = loss_f(pred_semantic, semantic)
                    loss += loss_value
                    loss_dict[name][mode] += loss_value.item()
                # if we're training, do backprop
                if mode == 'train':
                    loss.backward()
                    optimizer.step()
    
    # save loss values for this epoch
    train_loss, valid_loss = 0, 0
    for name, ldict in loss_dict.items():
        with open(os.path.join(output_dir, 'loss', f'{name}.csv'), 'a') as f:
            tval, vval = ldict['train'], ldict['valid']
            train_loss += tval
            valid_loss += vval
            f.writelines(['{:d},{:.5f},{:.5f}'.format(epoch, tval, vval)])
    pbar.set_postfix({'train' : train_loss, 'valid' : valid_loss})

    if cfg.EARLY_STOP:
        if early_stop.stop_early(valid_loss):
            stop_early = True
    else:
        stop_early = False
        
    # save the model checkpoint.
    if (epoch == 1) or (epoch % cfg.SAVE_EVERY == 0) or stop_early:
        state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict": model.state_dict(),
            "epoch": epoch
        }
        save_checkpoint(
            os.path.join(output_dir, "chkpt", f"model_{epoch}.pth"),
            state
        )

    # Make sure everything looks normal. 
    if epoch % cfg.IMAGE_EVERY == 0:
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
        ax0.imshow(pred_vf[0][0].detach().cpu())
        ax0.axis('off')
        ax1.imshow(pred_vf[0][1].detach().cpu())
        ax1.axis('off')
        ax2.imshow(image[0][0].detach().cpu())
        ax2.axis('off')
        ax3.imshow(pred_semantic[0][0].detach().cpu())
        ax3.axis('off')
        plt.savefig(
            os.path.join(output_dir, "img", f"image_{epoch}.png"), dpi = 400
        )
        plt.close()
    if stop_early:
        break

exit(0)
