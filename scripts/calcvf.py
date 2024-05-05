#!/usr/bin/env python

import os
import sys
import argparse

import torch
import numpy
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, "../torchvf/")
from dataloaders import *
from numerics import *


def main(**kwargs):
    args = argparse.Namespace(**kwargs)
    vf_delimiter = args.vf_delim
    # setup the device
    if torch.cuda.is_available() and not args.force_cpu:
        _device_str = 'cuda'
    else:
        _device_str = 'cpu'
    _device = torch.device('cuda')
    # make the dataset
    phead, ptail = os.path.split(args.data_dir)
    dset = BPCIS(phead, split=ptail, transforms=None)
    dload = DataLoader(dset, batch_size=1, drop_last=False, shuffle=False)
    save_dir = dset.split_dir
    for step, (image, mask) in enumerate(tqdm(dload)):
        image = image.to(_device).float()
        mask  = mask.to(_device).float()
        _, _, H, W = image.shape
        # calculate vector field
        vf = compute_vector_field(mask[0],
                                  args.kernel_size, alpha=args.alpha,
                                  device=_device_str)
        vf = vf.cpu().numpy().astype(numpy.float32)
        # save it
        old_fname = dset.imgs[step]
        new_vf_name = os.path.basename(old_fname).replace(
            ".tif", f"{vf_delimiter}.npy"
        )
        new_vf_path = os.path.join(save_dir, new_vf_name)
        numpy.save(new_vf_path, vf)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, required=True)
    parser.add_argument("-v", "--vf-delim", type=str, required=True)
    parser.add_argument("-a", "--alpha", type=int, default=10)
    parser.add_argument("-k", "--kernel-size", type=int, default=11)
    parser.add_argument("-fc", "--force-cpu", action="store_true", default=False)
    args = parser.parse_args()
    ec = main(**vars(args))
    exit(ec)
