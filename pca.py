import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sns.set()

class PCAReductor():

    def __init__(self, n_components, bl_position=273):
        self.bl_position = bl_position
        self.n_components = n_components
        self.pca = PCA(n_components)

    def perform(self, X):
        # Remember all black listed.
        self.bl_mask = np.empty((X.shape[0]), dtype=bool)
        for i in range(X.shape[0]):
            sample = X[i]
            if sample[self.bl_position]:
                self.bl_mask[i] = True
            else:
                self.bl_mask[i] = False

        # Dimensionality reduction.
        self.X = self.pca.fit_transform(X)

    def plot_data(self):
        if self.X is None:
            raise RuntimeError('There is no data to plot.')

        if self.n_components != 2:
            raise RuntimeError('Cannot plot if n_components is not 2.')

        # Coordinates for all data.
        x_axis_data = self.X[~self.bl_mask,0]
        y_axis_data = self.X[~self.bl_mask,1]

        # Coordinates for black listed.
        x_axis_black_listed = self.X[self.bl_mask,0]
        y_axis_black_listed = self.X[self.bl_mask,1] 

        # Plot.
        plt.scatter(x_axis_data, y_axis_data, color='r', label='Not blacklisted')
        plt.scatter(x_axis_black_listed, y_axis_black_listed, color='k', edgecolors='k', label='Blacklisted')
        plt.legend()
        plt.show()