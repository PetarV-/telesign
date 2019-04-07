import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import numpy as np

# TODO: Sigmoid/Tanh and log loss instead of MSE
class Autoencoder(nn.Module):
    def __init__(self, nb_in_features, nb_latent_features):
        super(Autoencoder, self).__init__()

        nb_mid_features = (nb_in_features + nb_latent_features) // 2

        # Experiment with batch norm
        # self.batch_norm = nn.BatchNorm1d(nb_in_features)

        self.encoder = nn.Sequential(
            nn.Linear(nb_in_features, nb_mid_features),
            nn.LeakyReLU(inplace=True),
            nn.Linear(nb_mid_features, nb_latent_features),
            nn.LeakyReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(nb_latent_features, nb_mid_features),
            nn.LeakyReLU(inplace=True),
            nn.Linear(nb_mid_features, nb_in_features)
        )
    
    def forward(self, X):
        # X_norm = self.batch_norm(X)
        L = self.encoder(X) # skip batch norm
        rec = self.decoder(L)
        return (X, L, rec)

# try 3 variants:
# mean na mean (ae-norm.pkl)
# original na batchnorm (ovo je radilo pre) (ae-batch2orig.pkl)
# batchnorm na batchnorm (ae-batch.pkl)

# A simple subclass of Dataset for the Loader
class SimpleDataset(Dataset):
    def __init__(self, X):
        self.X = X
        
    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]

# A driver class for the autoencoder
class AutoencoderDriver:
    def __init__(self, nb_in_features, nb_latent_features,
                 nb_epochs=100, batch_size=64):
        self.nb_epochs = nb_epochs 
        self.batch_size = batch_size
        self.nb_in_features = nb_in_features
        self.nb_latent_features = nb_latent_features

        self.autoencoder = Autoencoder(nb_in_features, nb_latent_features)
        self.autoencoder.double()
        self.autoencoder.cuda()

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5)

        print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Train on given data (shape is nb_examples X nb_in_features)
    def run(self, X):
        self.nb_examples = X.shape[0]
        assert(X.shape[1] == self.nb_in_features)

        dataset = SimpleDataset(X)

        # Train
        print("Training")
        self.autoencoder.train()
        for epoch in range(self.nb_epochs):
            epoch_loss = 0
            t_start = time.time()
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)  
            for batch in loader:
                batch = batch.cuda()
                self.optimizer.zero_grad()
                X_norm, _, rec = self.autoencoder(batch)

                # Backprop
                loss = self.loss_function(batch, rec)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            t_end = time.time()
            nb_batches = len(loader)
            if True:
                print('Epoch: {:04d}/{:04d}'.format(epoch+1, self.nb_epochs),
                      '|Loss: {:.6f}'.format(epoch_loss / nb_batches),
                      '|Time: {:.2f}s'.format(t_end - t_start))

        self.autoencoder.eval()

        # Print results
        x_torch = torch.tensor(X) 
        x_torch = x_torch.cuda()
        _, L_torch, _ = self.autoencoder(x_torch)
        L = L_torch.data.cpu().numpy()
        print(X.shape, ' ---> ', L.shape)
        print(L)

        with open('ae-norm-newdata.pkl', 'wb') as lt:
            pickle.dump(L, lt)