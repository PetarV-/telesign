import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

class SemiDataset(Dataset):
    # X: 9872 x ~2444 or 9872 x 100
    # Y: 9872 x 1
    # split: 'all', 'train', 'test', 'unlabeled'
    def __init__(self, use_latent = False, split = 'all', is_graph = False, test_ratio = 0.2):
        if use_latent:
            with open('ae-norm-newdata.pkl', 'rb') as X_file:
                X = pickle.load(X_file)
        else:
            with open('fts1.pkl', 'rb') as X_file:
                X = pickle.load(X_file)
        print("X shape: ", X.shape)
        self.nb_examples = X.shape[0]
        self.nb_features = X.shape[1]

        with open('labels.pkl', 'rb') as Y_file:
            Y = pickle.load(Y_file)
            Y = np.reshape(Y, (-1, 1))
            # [0, 1, 2, 3, 4 = unknown]

        # check 
        print("Y shape: ", Y.shape)
        
        # SPLIT
        if split == 'all':
            self.X = X 
            self.Y = Y 

        # SHUFFLE
        #perm = np.random.permutation(self.nb_examples)
        #X = X[perm]
        #Y = Y[perm]
        
        idx = np.reshape((Y == 4), -1)

        # graph stuff
        if is_graph:
            rng = np.arange(self.nb_examples)
            rng = rng[~idx]

        X_unlabeled = X[idx]
        Y_unlabeled = Y[idx]
        nb_unlabeled = X_unlabeled.shape[0]

        X_labeled = X[~idx]
        Y_labeled = Y[~idx]
        nb_labeled = X_labeled.shape[0]

        # graph stuff
        if is_graph:
            idx_X_train, idx_X_test, _, _ = train_test_split(rng, rng, test_size=0.2, stratify=Y_labeled)
            self.mask_train = torch.zeros(self.nb_examples)
            self.mask_train[idx_X_train] = 1
            self.mask_test = torch.zeros(self.nb_examples)
            self.mask_test[idx_X_test] = 1
            self.mask_train.cuda()
            self.mask_test.cuda()
        elif split != 'all':
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