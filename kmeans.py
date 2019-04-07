import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
import pickle 

# K Means predictor
class KMeansPredictor():

    def __init__(self, n_clusters, bl_position=273):
        self.n_clusters = n_clusters
        self.bl_position = bl_position
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=None)

    def perform(self, X, X_orig=None):
        if X_orig is None:
            X_orig = X
        # Remember all black listed.
        self.bl_mask = np.empty((X_orig.shape[0]), dtype=bool)
        total = 0
        for i in range(X_orig.shape[0]):
            sample = X_orig[i]
            if sample[self.bl_position]:
                self.bl_mask[i] = True
                total += 1
            else:
                self.bl_mask[i] = False
        
        print(total)

        # Perform kmeans.
        self.labels = self.kmeans.fit_predict(X)

        # Save 
        with open('labels.pkl', 'wb') as lb:
            pickle.dump(self.labels, lb)

    def feed_new_data(self, X):
        # Remember if new data is black listed.
        for i in range(X.shape[0]):
            sample = X[i]
            if sample[self.bl_position]:
                self.bl_mask[i].append(True)
            else:
                self.bl_mask[i].append(False)

        # Prediction for new data.
        new_labels = kmeans.predict(X)
        self.labels.append(new_labels)

    def black_listed_distribution(self):
        # Get number of black_listed in all clusters.
        bl_clusters = np.zeros(self.n_clusters, dtype=int)
        all_counts = np.zeros(self.n_clusters, dtype=int)
        bl_num = len([bm for bm in self.bl_mask if bm == True])

        for i in range(len(self.labels)):
            all_counts[self.labels[i]] += 1
            if self.bl_mask[i]:
                cluster = self.labels[i]
                bl_clusters[cluster] += 1
        
        print(all_counts)
        print(" (all counts)")

        # Sort descending all not 0 values.
        bl_clusters = [c for c in bl_clusters if c != 0]
        bl_clusters = np.sort(bl_clusters)
        bl_clusters = bl_clusters[::-1]

        # Print distribution.
        print('Black listed phone nums appear in {} clusters.'.format(len(bl_clusters)))
        print('Black listed distribution:')
        for i in range(len(bl_clusters)):
            perc = bl_clusters[i] / bl_num
            print('Cluster {}: {} percent of all blacklisted ({})'.format(i, perc, bl_clusters[i]))