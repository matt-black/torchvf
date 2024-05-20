import os

import numpy
import torch

from scipy.io import loadmat
from scipy.ndimage import label
from skimage.segmentation import expand_labels

class RgbDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, directory,
                 vf=False, vf_delimiter='vf.npy',
                 transforms=None, copy=None, remove=None,
                 expand=0):
        self._dir = directory
        # get all the files
        n_mats = len([f for f in os.listdir(self._dir)
                      if f.endswith('mat')])
        self._nums = [self._get_im_number(f) for f
                      in os.listdir(self._dir)
                      if f.endswith('mat')]
        self._n = n_mats
        # save vector field related properties
        self._vf = vf
        self._vf_delim = vf_delimiter
        # save transforms
        self._transforms = transforms
        # save copy/remove (NOTE: currently does nothing)
        self._copy = copy
        self._remove = remove
        # save expand
        self._expand = expand

    def _get_im_number(self, fname):
        return int(fname[2:5])

    def __getitem__(self, idx):
        fname = "im{:03d}.mat".format(self._nums[idx])
        fpath = os.path.join(self._dir, fname)
        # load mat
        mat = loadmat(fpath)
        rgb = mat['im']
        msk, _ = label(mat['labs'] > 0)
        if self._expand > 0:
            msk = expand_labels(msk, self._expand)
        msk = msk[numpy.newaxis,...]
        # load vector field, if applicable
        if self._vf:
            vf_name = fname.replace('.mat', "_{:s}.npy".format(self._vf_delim))
            vf = numpy.load(os.path.join(self._dir, vf_name))
            if self._transforms is not None:
                rgb, msk, vf = self._transforms(rgb, msk, vf)
            return rgb, vf, msk
        else:
            return rgb, msk
        
    def __len__(self):
        return self._n
