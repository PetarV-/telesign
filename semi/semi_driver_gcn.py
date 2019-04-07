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

class SemiDriverGcn:
    def __init__(self, model, nb_features, nb_classes, nb_epochs=100, batch_size=64,
                 use_latent = False):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.nb_epochs = 200
        self.batch_size = batch_size
        self.use_latent = use_latent

        with open('adj.pkl', 'rb') as ff:
            self.adj = pickle.load(ff)
        sums = np.sum(self.adj, 1)
        self.adj = self.adj / sums
        self.adj = torch.tensor(self.adj).cuda()

        self.model = model 
        self.model.double()
        self.model.cuda()

        self.class_weights = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0]).double().cuda()
        self.cross_entropy = nn.CrossEntropyLoss(weight = self.class_weights, reduction='none')
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-5)

        print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def run_test(self, split):
        print("Testing on \'{}\' dataset".format(split))
        dataset = SemiDataset(self.use_latent, split, is_graph = True)
        print("Dataset len: {}".format(len(dataset)))
        self.model.eval()
        
        test_loss = 0
        nb_examples = 0
        nb_correct = 0
        all_logits = []
        t_start = time.time()
        X = torch.tensor(dataset.X).cuda() 
        Y = torch.tensor(dataset.Y).cuda()
        out = self.model(X, self.adj)

        tmp = self.cross_entropy(out, Y.view(-1))
        tmp *= dataset.mask_test
        loss = torch.sum(tmp) / torch.sum(dataset.mask_test)

        test_loss = loss.item()
        _, predicted = torch.max(out.data, 1)
        all_logits.append(out.detach().cpu().numpy())
        predicted = predicted.view((-1, 1))
        #print(predicted)
        #print(batch_Y)

        examples = Y.size(0)
        nb_examples += examples

        pred_flags = (predicted == Y.data).double()
        #print(pred_flags.shape)
        #print("!")
        #print(dataset.mask_test.shape)
        tmp = pred_flags.view(-1) * dataset.mask_test
        #print(tmp.shape)
        tmp = torch.sum(tmp) / torch.sum(dataset.mask_test)
        acc = tmp

        #correct = int(tmp.sum())
        #nb_correct += correct
        #print ('Batch acc: {}'.format(float(correct)/examples))

        t_end = time.time()
        #print(acc)
        #print(acc.shape)
        #acc = float(nb_correct) / nb_examples
        print('Accuracy : {:.4f}'.format(acc),
            '|Loss: {:.6f}'.format(test_loss),
            '|Time: {:.2f}s'.format(t_end - t_start))
        print()
        return np.vstack(all_logits)

    def run(self):
        # Train
        print("Training")
        dataset = SemiDataset(self.use_latent, 'all', is_graph = True)


        self.model.train()
        for epoch in range(self.nb_epochs):
            epoch_loss = 0
            t_start = time.time()
            X = torch.tensor(dataset.X).cuda() 
            Y = torch.tensor(dataset.Y).cuda()
            self.optimizer.zero_grad()

            #print(X.shape)
            #print(self.adj.shape)

            out = self.model(X, self.adj)

            tmp = self.cross_entropy(out, Y.view(-1))
            tmp = tmp.double().cuda()
            #print(str(tmp))
            #print(str(dataset.mask_train))
            tmp *= dataset.mask_train
            loss = torch.sum(tmp) / torch.sum(dataset.mask_train)
            
            loss.backward()
            self.optimizer.step()
            loss = loss.item()
            t_end = time.time()
            if True:
                print('Epoch: {:04d}/{:04d}'.format(epoch+1, self.nb_epochs),
                      '|Loss: {:.6f}'.format(loss),
                      '|Time: {:.2f}s'.format(t_end - t_start))

        # Test 
        logits = self.run_test('all')
        row_sums = np.sum(np.exp(logits), axis=1)
        probs = np.exp(logits) / np.reshape(row_sums, (-1, 1))
        #print(probs.shape)
        #print(probs[:10, :])
        #print()

        np.set_printoptions(suppress=True)
        # POST PROCESSING
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        idxs = (entropy > 1.52)
        samo_true = [z for z in idxs if z == True]
        print(len(samo_true))
        #for i in range(9872):
        #    if not idxs[i]:
        #        continue
        #    print('Debaggg')
        #    print(sorted(list(probs[i])))
        #    input()
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
            