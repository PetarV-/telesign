import pickle
import numpy

from tsne import TSNEReductor
from pca import PCAReductor

with open('fts.pkl', 'rb') as features_file:
    features = pickle.load(features_file)
    print(features.shape)

ts = TSNEReductor(2, bl_position=273)

with open('ae-norm-newdata.pkl', 'rb') as latent_file:
    latent = pickle.load(latent_file)
    #for col in range(latent.shape[1]):
    #    column = latent[:, col]
    #    mean = numpy.mean(column)
    #    std = numpy.std(column)
    #    latent[:, col] = (column - mean) / std
    ts.perform(latent, features)
    ts.plot_data()

