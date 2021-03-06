import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import numpy as np
from semi_dataset import SemiDataset
from mlp import MLP
import matplotlib.pyplot as plt

# Driver class for MLP
class SemiDriver:
    def __init__(self, model, nb_features, nb_classes, nb_epochs=100, batch_size=64,
                 use_latent = False, test_ratio = 0.2):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.nb_epochs = nb_epochs 
        self.batch_size = batch_size
        self.use_latent = use_latent
        self.test_ratio = test_ratio

        self.model = model 
        self.model.double()
        self.model.cuda()

        self.class_weights = torch.Tensor([1.0, 1.0, 0.002, 0.002, 1.0]).double().cuda()
        self.cross_entropy = nn.CrossEntropyLoss(weight = self.class_weights)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-5)

        print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def run_test(self, split):
        print("Testing on \'{}\' dataset".format(split))
        dataset = SemiDataset(self.use_latent, split, test_ratio=self.test_ratio)
        print("Dataset len: {}".format(len(dataset)))
        self.model.eval()
        
        test_loss = 0
        nb_examples = 0
        nb_correct = 0
        all_logits = []
        t_start = time.time()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)      
        for batch_X, batch_Y in loader:
            # Get the output and prediction
            batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()
            out = self.model(batch_X)
            loss = self.cross_entropy(out, batch_Y.view(-1))
            test_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            all_logits.append(out.detach().cpu().numpy())
            predicted = predicted.view((-1, 1))

            # Calculate accuracy
            examples = batch_Y.size(0)
            nb_examples += examples
            correct = int((predicted == batch_Y.data).sum())
            nb_correct += correct

        t_end = time.time()
        nb_batches = len(loader)
        acc = float(nb_correct) / nb_examples
        print('Accuracy : {:.4f}'.format(acc),
            '|Loss: {:.6f}'.format(test_loss / nb_batches),
            '|Time: {:.2f}s'.format(t_end - t_start))
        print()
        return np.vstack(all_logits), acc

    def run(self):
        # Train
        print("Training")
        train_dataset = SemiDataset(self.use_latent, 'train', test_ratio = self.test_ratio)
        self.model.train()
        for epoch in range(self.nb_epochs):
            epoch_loss = 0
            t_start = time.time()
            loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  
            for batch_X, batch_Y in loader:
                # Get the output
                batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()
                self.optimizer.zero_grad()
                out = self.model(batch_X)
                loss = self.cross_entropy(out, batch_Y.view(-1))

                # Backprop
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            t_end = time.time()
            nb_batches = len(loader)
            if epoch % 10 == 9:
                print('Epoch: {:04d}/{:04d}'.format(epoch+1, self.nb_epochs),
                      '|Loss: {:.6f}'.format(epoch_loss / nb_batches),
                      '|Time: {:.2f}s'.format(t_end - t_start))

        # Test 
        self.run_test('train')
        _, test_accuracy = self.run_test('test')
        
        logits, _ = self.run_test('all')
        row_sums = np.sum(np.exp(logits), axis=1)
        probs = np.exp(logits) / np.reshape(row_sums, (-1, 1))

        np.set_printoptions(suppress=True)

        # Post processing, thresh on entropy
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        idxs = (entropy > 1.52)
        probs[idxs, :4] = 0
        probs[idxs, 4] = 1 

        # Save probabilities
        with open('probs.pkl', 'wb') as lt:
            pickle.dump(probs, lt)

        only_true = [z for z in idxs if z == True]
        print(len(only_true))

        predictions = np.argmax(probs, axis=1)
        predictions[idxs] = 4
        
        predictions = np.reshape(predictions, (-1, 1))

        # Save predictions
        with open('predictions.pkl', 'wb') as lt:
            pickle.dump(predictions, lt)
        
        print("Done semi driver")
        return test_accuracy
            