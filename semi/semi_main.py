import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import numpy as np
from semi_driver import SemiDriver 
from mlp import MLP 

use_latent = False
nb_features = 100 if use_latent else 2444 #?
nb_classes = 5
model = MLP(nb_features, nb_classes)
driver = SemiDriver(model, nb_features, nb_classes, 
                    nb_epochs=100, batch_size=64, 
                    use_latent=use_latent)
driver.run() # let's go