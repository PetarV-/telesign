import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PCAReductor():

    def __init__(self, n_components, bl_position):
        self.bl_position = bl_position
        self.n_components = n_components
        self.pca = PCA(n_components)

    def perform(self, X):
        # Remember all black listed.
        self.bl_indices = []
        for i in range(X.shape[0]):
            sample = X[i]
            if sample[bl_position]:
                self.bl_indices.append(i)
        self.bl_indices = np.asarray(self.bl_indices)

        # Dimensionality reduction.
        self.X = pca.fit_transform(X)

    def plot_data(self):
        if self.X is None:
            raise RuntimeError('There is no data to plot.')

        if self.n_components != 2;
            raise RuntimeError('Cannot plot if n_components is not 2.')

        # Coordinates for all data.
        x_axis_data = self.X[~self.bl_indices,0]
        y_axis_data = self.X[~self.bl_indices,1]

        # Coordinates for black listed.
        x_axis_black_listed = self.X[self.bl_indices,0]
        y_axis_black_listed = self.X[self.bl_indices,1] 

        # Plot.
        plt.plot(x_axis_data, y_axis_data, 'r', label='All data')
        plt.plot(x_axis_black_listed, x_axis_black_listed, 'g', label='Black listed')
        plt.legend()
        plt.show()