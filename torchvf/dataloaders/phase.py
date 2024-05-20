import os

import numpy
import torch
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import label
from skimage.util import img_as_ubyte
from skimage.segmentation import expand_labels

def PhaseDataset(data_dir,
                 vf=False, vf_name='vf.npy',
                 transforms=None, copy=None, remove=None,
                 expand=0, set_type='mat'):
    """
    """
    if not vf_name.endswith('.npy'):
        vf_name = vf_name + '.npy'
    if set_type == 'mat':
        return PhaseDatasetMat(data_dir, vf, vf_name,
                               transforms, copy, remove,
                               expand)
    elif set_type == 'img':
        return PhaseDatasetImg(data_dir, vf, vf_name,
                               transforms, copy, remove,
                               expand)
    else:
        raise ValueError('invalid set_type')


class PhaseDatasetMat(torch.utils.data.Dataset):
    """
    """
    def __init__(self, data_dir,
                 vf=False, vf_name='vf.npy',
                 transforms=None, copy=None, remove=None,
                 expand=0):
        self._fldrs = _find_dirs_with_mat(data_dir)
        # vector field handling
        self._vf = vf
        if vf_name.endswith('.npy'):
            self._vf_name = vf_name
        else:
            self._vf_name = vf_name + '.npy'
        self.transforms = transforms
        if copy is not None:
            for i in range(len(copy)):
                self._fldrs.append(self._fldrs[i])
        if remove is not None:
            self._fldrs = [f for i, f in enumerate(self._fldrs)
                           if i not in remove]
        self._expand = expand

    def __getitem__(self, idx):
        image, mask = self._get_image_mask(idx)
        if self._vf:
            vf = self._get_vf(idx)
            if self.transforms is not None:
                image, mask, vf = self.transforms(image, mask, vf)
            return image, vf, mask
        else:
            if self.transforms is not None:
                image, mask, _ = self.transforms(image, mask)            
            return image, mask
        
    def _get_vf(self, idx):
        fldr = self._fldrs[idx]
        fpath = os.path.join(fldr, self._vf_name)
        vf = numpy.load(fpath)
        return vf

    def _get_image_mask(self, idx):
        mat = self._get_mat(idx)
        # get phase image & convert to 8-bit
        phs = (mat["im"][0,...]).astype(numpy.uint16)
        phs = img_as_ubyte(phs)
        # get mask
        msk, _ = label(mat["labs"] == 1)
        if self._expand > 0:
            msk = expand_labels(msk, self._expand)
        return phs[numpy.newaxis,...], msk[numpy.newaxis,...]

    def __len__(self):
        return len(self._fldrs)

    def _get_mat(self, idx):
        fldr = self._fldrs[idx]
        return loadmat(os.path.join(fldr, 'LabelMask.mat'))


def _find_dirs_with_mat(root_dir):
    fldrs = []
    for this_fldr, subdirs, files in os.walk(root_dir):
        for fname in files:
            if fname == "LabelMask.mat":
                fldrs.append(this_fldr)
    return fldrs


class PhaseDatasetImg(torch.utils.data.Dataset):
    """
    """
    def __init__(self, data_dir,
                 vf=False, vf_name='vf.npy',
                 transforms=None, copy=None, remove=None,
                 expand=0):
        self._fldrs = _find_dirs_with_mat(data_dir)
        # vector field handling
        self._vf = vf
        if vf_name.endswith('.npy'):
            self._vf_name = vf_name
        else:
            self._vf_name = vf_name + '.npy'
        self.transforms = transforms
        if copy is not None:
            for i in range(len(copy)):
                self._fldrs.append(self._fldrs[i])
        if remove is not None:
            self._fldrs = [f for i, f in enumerate(self._fldrs)
                           if i not in remove]
        self._expand = expand

    def __getitem__(self, idx):
        image = self._get_phase(idx)
        mask = self._get_mask(idx)
        if self._vf:
            vf = self._get_vf(idx)
            if self.transforms is not None:
                image, mask, vf = self.transforms(image, mask, vf)
            return image, vf, mask
        else:
            if self.transforms is not None:
                image, mask, _ = self.transforms(image, mask)
            return image, mask

    def __len__(self):
        return len(self._fldrs)

    def _get_phase(self, idx):
        fldr = self._fldrs[idx]
        try:
            phs_im = Image.open(os.path.join(fldr, "phase_raw.tif"))
        except:
            phs_im = Image.open(os.path.join(fldr, "phase_raw.png"))
        phs = numpy.asarray(phs_im).byteswap().newbyteorder()
        return phs.astype(int)

    def _get_mask(self, idx):
        fldr = self._fldrs[idx]
        msk_png = Image.open(os.path.join(fldr, "raw_mask.png"))
        _, _, _, msk = msk_png.split()
        msk, _ = label(numpy.asarray(msk) > 0)
        if self._expand > 0:
            return expand_labels(msk, self._expand)
        else:
            return msk

    def _get_vf(self, idx):
        fldr = self._fldrs[idx]
        fpath = os.path.join(fldr, self._vf_name)
        vf = numpy.load(fpath)
        return vf


def _find_dirs_with_mask(root_dir):
    fldrs = []
    for this_fldr, subdirs, files in os.walk(root_dir):
        for fname in files:
            if fname == 'raw_mask.png':
                fldrs.append(this_fldr)
    return fldrs
