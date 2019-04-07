import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import numpy as np

class MLP(nn.Module):
    def __init__(self, nb_features, nb_classes):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(nb_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, nb_classes)
        )
    
    def forward(self, X):
        return self.mlp(X)