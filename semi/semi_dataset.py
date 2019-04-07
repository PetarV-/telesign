import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import numpy as np

class SemiDataset(Dataset):
    # X: 9872 x ~2444 or 9872 x 100
    # Y: 9872 x 1
    # split: 'all', 'train', 'test', 'unlabeled'
    def __init__(self, use_latent = False, split = 'all'):
        if use_latent:
            with open('ae-norm-newdata.pkl', 'rb') as X_file:
                self.X = pickle.load(X_file)
        else:
            with open('fts1.pkl', 'rb') as X_file:
                self.X = pickle.load(X_file)
        print("X shape: ", self.X.shape)
        self.nb_examples = self.X.shape[0]
        self.nb_features = self.X.shape[1]

        with open('labels.pkl', 'rb') as Y_file:
            self.Y = pickle.load(Y_file)
            self.Y = np.reshape(self.Y, (-1, 1))
            # [0, 1, 2, 3, 4 = unknown]
        
        # SHUFFLE
        self.X = np.random.shuffle(self.X)
        self.Y = np.random.shuffle(self.Y)
        
        # SPLIT
        if split == 'all':
            return
        
        idx = np.reshape((self.Y == 4), -1)

        X_unlabeled = X[idx]
        Y_unlabeled = Y[idx]
        nb_unlabeled = X_unlabeled.shape[0]

        X_labeled = X[~idx]
        Y_labeled = X[~idx]
        nb_labeled = X_labeled.shape[0]

        train_ratio = 0.8

        nb_train = int(nb_labeled * train_ratio)
        nb_test = nb_labeled - nb_train 

        X_train = X[:nb_train]
        Y_train = Y[:nb_train]

        X_test = X[nb_train:]
        Y_test = Y[nb_train:]
        
        if split == 'unlabeled':
            self.X = X_unlabeled
            self.Y = Y_unlabeled
        elif split == 'train':
            self.X = X_train 
            self.Y = Y_train 
        elif split == 'test':
            self.X = X_test 
            self.Y = Y_test 
        else:
            raise RuntimeError('unknown split: \'{}\''.format(split))
        
        
    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

    def __len__(self):
        return self.X.shape[0]