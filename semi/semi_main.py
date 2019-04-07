import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import numpy as np
from semi_driver import SemiDriver 
from mlp import MLP 
import matplotlib.pyplot as plt

use_latent = True
nb_features = 100 if use_latent else 2444 #?
nb_classes = 5
model = MLP(nb_features, nb_classes)
driver = SemiDriver(model, nb_features, nb_classes, 
                    nb_epochs=100, batch_size=64, 
                    use_latent=use_latent,
                    test_ratio = 0.2)
accuracy = driver.run() # let's go
#accuracies.append(accuracy)

#print(list(zip(test_ratios, accuracies)))

#import code 
#code.interact(local=locals())

#def moving_average(x, w):
#    return np.convolve(x, np.ones(w), 'same') / w#

#test_ratios = list(test_ratios)
#accuracies = moving_average(accuracies, 3)
#plt.plot(test_ratios, accuracies)
#plt.show()