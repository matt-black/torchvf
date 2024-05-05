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

##############################################################
# The Bacterial Phase Contrast for Instance Segmentation (BPCIS) 
# dataset that was provided along with the Omnipose paper. More
# information can be found here: 
# - Paper: https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2
# - GitHub: https://github.com/kevinjohncutler/omnipose
# - Papers with Code: https://paperswithcode.com/dataset/bpcis
# 
##############################################################

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch

import numpy as np 
import glob
import cv2
import os


class BPCIS(Dataset):
    """
    Dataloader for the BPCIS dataset. The link to download this dataset can be
    found on the Omnipose homepage here: 
        - http://www.cellpose.org/omnipose

    Args:
        data_dir (str): The location of the BPCIS dataset relative to your
            current directory.

        split (str): Options are:
            - "bact_fluor_train" (default)
            - "bact_fluor_test"
            - "bact_phase_train"
            - "bact_phase_test"
            - "worm_train"
            - "worm_test"

        vf (bool): If True, __getitem__ will return the vector fields (assuming
            they have been computed).

        vf_delimeter (str): The delimeter of the vector fields in storage. For
            example: "_vf" would assume the files containing the vector fields 
            are in the form "*_vf.npy".

    Returns:
        __getitem__: Return the image and mask respectively of index i. 

    """
    def __init__(self, data_dir, split = "bact_fluor_train",
                 vf = False, vf_delimiter = "_vf", transforms = None,
                 copy = None, remove = None):
        self.transforms = transforms 
        self.split_dir = os.path.join(data_dir, split)

        self.imgs, self.masks, self.vfs = [], [], []
        for root, _, files in os.walk(self.split_dir):
            msks = [f for f in files if f.endswith('_masks.tif')]
            imgs = [f[:-10]+".tif" for f in msks]
            vfs  = [i.replace(".tif", f"{vf_delimiter}.npy") for i in imgs]
            self.masks += [os.path.join(root, m) for m in msks]
            self.imgs += [os.path.join(root, i) for i in imgs]
            self.vfs += [os.path.join(root, "..", vf) for vf in vfs]
        if vf:
            self._getitem = self._get_image_mask_vf
        else:
            self._getitem = self._get_image_mask
        if copy is not None:
            for i in range(len(copy)):
                self.imgs.append(self.imgs[i])
                self.masks.append(self.masks[i])
                if vf:
                    self.vfs.append(self.vfs[i])
        if remove is not None:
            self.imgs  = [i for j, i in enumerate(self.imgs) if j not in remove]
            self.masks = [i for j, i in enumerate(self.masks) if j not in remove]
            if vf:
                self.vfs = [i for j, i in enumerate(self.vfs) if j not in remove]

    def _get_image_mask(self, idx):
        input_dir = self.imgs[idx]
        mask_dir  = self.masks[idx]

        image = cv2.imread(input_dir, cv2.IMREAD_GRAYSCALE).astype(
            np.float32
        )[None]
        mask  = cv2.imread(mask_dir,  cv2.IMREAD_UNCHANGED).astype(
            np.float32
        )[None]

        if self.transforms is not None:
            image, mask, _ = self.transforms(image, mask)

        return image, mask

    def _get_image_mask_vf(self, idx):
        input_dir = self.imgs[idx]
        mask_dir  = self.masks[idx]
        vf_dir    = self.vfs[idx]

        image = cv2.imread(input_dir, cv2.IMREAD_GRAYSCALE).astype(
            np.float32
        )[None]
        mask  = cv2.imread(mask_dir,  cv2.IMREAD_UNCHANGED).astype(
            np.float32
        )[None]
        vf    = np.load(vf_dir).astype(np.float32)
        if self.transforms is not None:
            image, mask, vf = self.transforms(image, mask, vf)
        return image, vf, mask

    def __getitem__(self, idx):
        return self._getitem(idx)

    def __len__(self):
        return len(self.imgs)
