import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import numpy as np
from semi_dataset import SemiDataset
from mlp import MLP

class SemiDriver:
    def __init__(self, model, nb_features, nb_classes, nb_epochs=100, batch_size=64,
                 use_latent = False):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.nb_epochs = nb_epochs 
        self.batch_size = batch_size
        self.use_latent = use_latent

        self.model = model 
        self.model.double()
        self.model.cuda()

        self.cross_entropy = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-5)

        print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def run_test(self, split):
        print("Testing on \'{}\' dataset".format(split))
        dataset = SemiDataset(self.use_latent, split)
        print("Dataset len: {}".format(len(dataset)))
        self.model.eval()
        
        test_loss = 0
        nb_examples = 0
        nb_correct = 0
        t_start = time.time()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)      
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()
            out = self.model(batch_X)
            loss = self.cross_entropy(out, batch_Y.view(-1))
            test_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.view((-1, 1))
            #print(predicted)
            #print(batch_Y)

            examples = batch_Y.size(0)
            nb_examples += examples
            correct = int((predicted == batch_Y.data).sum())
            nb_correct += correct
            #print ('Batch acc: {}'.format(float(correct)/examples))

        t_end = time.time()
        nb_batches = len(loader)
        acc = float(nb_correct) / nb_examples
        print('Accuracy : {:.4f}'.format(acc),
            '|Loss: {:.6f}'.format(test_loss / nb_batches),
            '|Time: {:.2f}s'.format(t_end - t_start))
        print()

    def run(self):
        # Train
        print("Training")
        train_dataset = SemiDataset(self.use_latent, 'train')
        self.model.train()
        for epoch in range(self.nb_epochs):
            epoch_loss = 0
            t_start = time.time()
            loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  
            for batch_X, batch_Y in loader:
                batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()
                self.optimizer.zero_grad()
                out = self.model(batch_X)
                #print(out)
                #print(batch_Y)
                #print(out.shape)
                #print(batch_Y.shape)
                loss = self.cross_entropy(out, batch_Y.view(-1))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            t_end = time.time()
            nb_batches = len(loader)
            if True:
                print('Epoch: {:04d}/{:04d}'.format(epoch+1, self.nb_epochs),
                      '|Loss: {:.6f}'.format(epoch_loss / nb_batches),
                      '|Time: {:.2f}s'.format(t_end - t_start))

        # Test 
        self.run_test('train')
        self.run_test('test')
            