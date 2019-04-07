from kmeans import KMeansPredictor
import numpy as np
import pickle

# KMeans main class

#nb_examples = 9872
#nb_features = 2427

km = KMeansPredictor(5)

with open('fts.pkl', 'rb') as features_file:
    features = pickle.load(features_file)
    print(features.shape)

with open('latent.pkl', 'rb') as latent_file:
    latent = pickle.load(latent_file)
    #for col in range(latent.shape[1]):
    #    column = latent[:, col]
    #    mean = numpy.mean(column)
    #    std = numpy.std(column)
    #    latent[:, col] = (column - mean) / std

    km.perform(latent, features)
    km.black_listed_distribution()