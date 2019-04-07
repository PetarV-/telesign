import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# A class for our semisupervised learning dataset
class SemiDataset(Dataset):
    # X shape will be 9872 x 2444 or 9872 x 100
    # Y shape will be 9872 x 1
    # split can take following values: 'all', 'train', 'test', 'unlabeled'
    def __init__(self, use_latent = False, split = 'all', is_graph = False, test_ratio = 0.2):
        if use_latent:
            # Load autoencoded latent vectors
            with open('ae-norm-newdata.pkl', 'rb') as X_file:
                X = pickle.load(X_file)
        else:
            # Load second iteration of full feature vectors
            with open('fts1.pkl', 'rb') as X_file:
                X = pickle.load(X_file)
        print("X shape: ", X.shape)
        self.nb_examples = X.shape[0]
        self.nb_features = X.shape[1]

        # Load labels picked from tSNE plot
        with open('labels.pkl', 'rb') as Y_file:
            Y = pickle.load(Y_file)
            Y = np.reshape(Y, (-1, 1))
            # [0, 1, 2, 3, 4 = unknown]
        
        # If we need all this is it
        if split == 'all':
            self.X = X 
            self.Y = Y 
        
        # Select those without labels
        idx = np.reshape((Y == 4), -1)

        # For graphs we don't have batches so we act a bit differently
        if is_graph:
            rng = np.arange(self.nb_examples)
            rng = rng[~idx]

        # Split labeled and unlabeled
        X_unlabeled = X[idx]
        Y_unlabeled = Y[idx]
        nb_unlabeled = X_unlabeled.shape[0]

        X_labeled = X[~idx]
        Y_labeled = Y[~idx]
        nb_labeled = X_labeled.shape[0]

        if is_graph:
            # Prepare masks
            idx_X_train, idx_X_test, _, _ = train_test_split(rng, rng, test_size=0.2, stratify=Y_labeled)
            self.mask_train = torch.zeros(self.nb_examples)
            self.mask_train[idx_X_train] = 1
            self.mask_test = torch.zeros(self.nb_examples)
            self.mask_test[idx_X_test] = 1
            self.mask_train = self.mask_train.double().cuda()
            self.mask_test = self.mask_test.double().cuda()
        elif split != 'all':
            # Prepare train and test subsets
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_labeled, Y_labeled, test_size=test_ratio, stratify=Y_labeled)

            nb_train = X_train.shape[0]
            nb_test = X_test.shape[0]

            print('Creating a dataset...')
            print('nb_train={} nb_test={} nb_labeled={} nb_unlab={} nb_examples={}'.format(
                nb_train, nb_test, nb_labeled, nb_unlabeled, self.nb_examples)) 
            
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