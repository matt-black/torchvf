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

import cupy
import numpy
import torch

import cuml
from sklearn.cluster import DBSCAN as DBSCAN_cpu
from sklearn.neighbors import NearestNeighbors as NN_cpu


def cluster(points: torch.Tensor, semantic: torch.Tensor,
            eps: bool = 2.2, min_samples: int = 15, snap_noise: bool = True):
    """
    Returns the instance segmentation given the the semantic segmentation and 
    the integrated euclidean semantic points. 

    Args:
        points (torch.Tensor): Of shape (D, N).
        semantic (torch.Tensor): The semantic segmentation of shape (1, H, W).
        eps (float): EPS value to use for DBSCAN. This is the neighborhood 'radius' from 
            a point we want to consider. 
        min_samples (int): min_samples value to use for DBSCAN.
        snap_noise (bool): force points detected as noise into their nearest cluster

    Returns:
        torch.Tensor: The instance segmentation of shape (1, H, W).

    """
    if (not points.is_cuda) or (not semantic.is_cuda):
        # inputs are CPU
        __cpu = True
        DBSCAN = DBSCAN_cpu
        NearestNeighbors = NN_cpu
    else:  # inputs are GPU
        __cpu = False
        DBSCAN = cuml.DBSCAN
        NearestNeighbors = cuml.neighbors.NearestNeighbors
        # need to move inputs to cupy
        points = cupy.from_dlpack(points)
        semantic = cupy.from_dlpack(semantic.int())
    # get data
    points = points.T
    if points.shape[0] == 0:
        return semantic
    # do DBSCAN clustering
    clustering = DBSCAN(
        eps=eps, min_samples=min_samples
    ).fit(points)

    if __cpu:
        clusters = torch.Tensor(clustering.labels_)
    else:
        clusters = torch.from_dlpack(clustering.labels_)

    if snap_noise:
        outliers_idx = clusters == -1
        if torch.any(outliers_idx):
            non_outliers_idx = ~outliers_idx

            # If there are ONLY outliers, then nothing was 
            # clustered and return the semantic segmentation. 
            if not torch.any(non_outliers_idx):
                return semantic
            # find outliers
            outliers     = points[outliers_idx]
            non_outliers = points[non_outliers_idx]
            # find nearest neighbors of each outlier to a non-outlier
            NN = NearestNeighbors(metric = "euclidean")
            NN.fit(cupy.from_dlpack(non_outliers))
            _, nn_idx = NN.kneighbors(outliers)
            # move stuff back to torch
            if __cpu:
                nn_idx = torch.Tensor(nn_idx).long()
            else:
                nn_idx = torch.from_dlpack(nn_idx).long()
            nearest_n = clusters[non_outliers_idx][nn_idx]
            values = torch.mode(nearest_n, dim = 1)[0]
            # Give the outliers the instance value of their nearest 
            # clustered neighbor.
            clusters[outliers_idx] = values
    
    if not __cpu:
        # need to move stuff back to pytorch
        points = torch.from_dlpack(points)
        semantic = torch.from_dlpack(semantic)
    # It sets first cluster to 0, be careful as this would
    # blend it into the background.
    instance_segmentation = semantic.clone().int()
    instance_segmentation[semantic.bool()] = clusters.int() + 1
    return instance_segmentation
