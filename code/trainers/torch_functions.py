
# MIT License

# Copyright (c) 2018 Nam Vo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""Metric learning functions.

Codes are modified from:
https://github.com/lugiavn/generalization-dml/blob/master/nams.py
"""

import numpy as np
import torch
import torchvision


def pairwise_distances(x, y=None):
  """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source:
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    """
  x_norm = (x**2).sum(1).view(-1, 1)
  if y is not None:
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
  else:
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

  dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
  # Ensure diagonal is zero if x=y
  # if y is None:
  #     dist = dist - torch.diag(dist.diag)
  return torch.clamp(dist, 0.0, np.inf)


class NormalizationLayer(torch.nn.Module):
  """Class for normalization layer."""
  def __init__(self, normalize_scale=1.0, learn_scale=True):
    super(NormalizationLayer, self).__init__()
    self.norm_s = float(normalize_scale)
    if learn_scale:
      self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

  def forward(self, x):
    features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
    return features
