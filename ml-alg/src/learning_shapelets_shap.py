from collections import OrderedDict
import warnings

import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import torch.nn as nn

class ShapeletSHAPWrapper(nn.Module):
    def __init__(self, linear_layer):
        super().__init__()
        self.linear = linear_layer

    def forward(self, x):
        # x: [n_samples, num_shapelets]
        x = x.unsqueeze(1)  # Add the channel dim back if needed: [n_samples, 1, num_shapelets]
        out = self.linear(x)  # Should work if the original model uses Conv1D or Linear
        out = torch.squeeze(out, 1)
        return out