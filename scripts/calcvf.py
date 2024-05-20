#!/usr/bin/env python

import os
import sys
import argparse

import torch
import numpy
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torchvf.dataloaders import *
from torchvf.numerics import *


def main_phase(**kwargs):
    args = argparse.Namespace(**kwargs)
    vf_name = args.vf_delim
    # setup the device
    if torch.cuda.is_available() and not args.force_cpu:
        _device_str = 'cuda'
    else:
        _device_str = 'cpu'
    _device = torch.device('cuda')
    # make the dataset
    dset = PhaseDataset(args.data_dir, vf=False, transforms=None,
                        copy=None, remove=None,
                        expand=args.expand,
                        set_type=args.phase_set_type)
    dload = DataLoader(
        dset, batch_size=1, drop_last=False, shuffle=False
    )
    # do the actual calculation
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
        fldr = dset._fldrs[step]
        new_vf_path = os.path.join(fldr, vf_name)
        numpy.save(new_vf_path, vf)
    return 0


def main_rgb(**kwargs):
    args = argparse.Namespace(**kwargs)
    vf_delimiter = args.vf_delim
    if torch.cuda.is_available() and not args.force_cpu:
        _device_str = 'cuda'
    else:
        _device_str = 'cpu'
    _device = torch.device('cuda')
    # make the dataset
    dset = RgbDataset(args.data_dir, vf=False, transforms=None,
                      copy=None, remove=None,
                      expand=args.expand)
    dload = DataLoader(
        dset, batch_size=1, drop_last=False, shuffle=False
    )
    # do the actual calculation
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
        num = dset._nums[step]
        vf_fname = "im{:03d}_".format(num) + f"{vf_delimiter}"
        numpy.save(os.path.join(args.data_dir, vf_fname), vf)
    return 0


def main_bpcis(**kwargs):
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
            ".tif", f"{vf_delimiter}"
        )
        new_vf_path = os.path.join(save_dir, new_vf_name)
        numpy.save(new_vf_path, vf)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, required=True)
    parser.add_argument("-t", "--data-type", type=str, required=True,
                        choices=["bpcis", "phase", "rgb"])
    parser.add_argument("-v", "--vf-delim", type=str, required=True)
    parser.add_argument("-a", "--alpha", type=int, default=10)
    parser.add_argument("-k", "--kernel-size", type=int, default=11)
    parser.add_argument("-e", "--expand", type=int, default=1)
    parser.add_argument("-pst", "--phase-set-type", type=str,
                        choices=["mat", "img"])
    parser.add_argument("-fc", "--force-cpu", action="store_true", default=False)
    args = parser.parse_args()
    if args.data_type == 'bpcis':
        ec = main_bpcis(**vars(args))
    elif args.data_type == 'phase':
        ec = main_phase(**vars(args))
    else:
        ec = main_rgb(**vars(args))
    exit(ec)
