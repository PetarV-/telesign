from autoencoder import AutoencoderDriver
import numpy as np
import pickle

#nb_examples = 9872
#nb_features = 2427

with open('fts1.pkl', 'rb') as features_file:
    features = pickle.load(features_file)
    nb_examples = features.shape[0]
    nb_features = features.shape[1]
    print(nb_examples, nb_features)
    to_normalize = [1344, 1345, 1346, 1347, 1349, 1350, 2420, 2421, 2422, 2423, 2425, 2426]
    for col in to_normalize:
        column = features[:, col]
        mean = np.mean(column)
        std = np.std(column)
        features[:, col] = (column - mean) / std
    
    ad = AutoencoderDriver(nb_in_features=nb_features, 
        nb_latent_features=100, batch_size=64, nb_epochs=100)

    ad.run(features)