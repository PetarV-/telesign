import pickle
import numpy

with open('fts.pkl', 'rb') as features_file:
    features = pickle.load(features_file)
    print(features.shape)
    to_normalize = [1344, 1345, 1346, 1347, 1349, 1350, 2420, 2421, 2422, 2423, 2425, 2426]
    for col in to_normalize:
        column = features[:, col]
        print(column)
        mean = numpy.mean(column)
        std = numpy.std(column)
        column = (column - mean) / std
        print(column)
