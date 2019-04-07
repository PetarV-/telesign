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
        all_logits = []
        t_start = time.time()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)      
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.cuda(), batch_Y.cuda()
            out = self.model(batch_X)
            loss = self.cross_entropy(out, batch_Y.view(-1))
            test_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            all_logits.append(out.detach().cpu().numpy())
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
        return np.vstack(all_logits)

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
            if epoch % 10 == 9:
                print('Epoch: {:04d}/{:04d}'.format(epoch+1, self.nb_epochs),
                      '|Loss: {:.6f}'.format(epoch_loss / nb_batches),
                      '|Time: {:.2f}s'.format(t_end - t_start))

        # Test 
        self.run_test('train')
        self.run_test('test')
        
        logits = self.run_test('all')
        row_sums = np.sum(np.exp(logits), axis=1)
        probs = np.exp(logits) / np.reshape(row_sums, (-1, 1))
        #print(probs.shape)
        #print(probs[:10, :])
        #print()

        # POST PROCESSING
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        idxs = (entropy > 1.5)
        for i in range(1000):
            if not idxs[i]:
                continue
            print('Debaggg')
            print(probs[i])
            input()
        predictions = np.argmax(probs, axis=1)
        predictions[idxs] = 4
        print(predictions.shape)
        print(predictions)
        #return


        #tmp = sorted(list(entropy))
        #plt.hist(tmp, bins=100)
        #plt.show()
        #return



        #np_predictions = np.asarray(predictions)
        
        predictions = np.reshape(predictions, (-1, 1))

        #print(np_predictions.shape)
        #print(np_predictions)

        with open('predictions.pkl', 'wb') as lt:
            pickle.dump(predictions, lt)
        
        print("E stigao si do kraja")
            