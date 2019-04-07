import pickle
import numpy

from tsne import TSNEReductor
from pca import PCAReductor

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

with open('fts.pkl', 'rb') as features_file:
    features = pickle.load(features_file)
    print(features.shape)
    to_normalize = [1344, 1345, 1346, 1347, 1349, 1350, 2420, 2421, 2422, 2423, 2425, 2426]
    for col in to_normalize:
        column = features[:, col]
        mean = numpy.mean(column)
        std = numpy.std(column)
        #features[:, col] = (column - mean) / std

diffs = []

for i in range(features.shape[0]):
    if features[i, 2425] - features[i, 1349] > 0:
        diffs.append(features[i, 2425] - features[i, 1349])

#n, bins, _ = plt.hist(diffs, 500)
#print(n)
#print(bins)
#plt.show()

#exit(0)

ts = TSNEReductor(2, bl_position=273)

ts.perform(features)
ts.plot_data()

