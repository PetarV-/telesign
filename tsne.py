import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sns.set()

class TSNEReductor():

    def __init__(self, n_components, bl_position=273):
        self.bl_position = bl_position
        self.sms_position = bl_position + 1
        self.mbl_position = bl_position - 7
        self.fxd_position = bl_position - 6
        self.tch_position = bl_position - 5
        self.ott_position = bl_position - 4
        self.prm_position = bl_position - 3
        self.tlf_position = bl_position - 2
        self.n_components = n_components
        self.tsne = TSNE(n_components)

    def perform(self, X):
        # Remember all black listed.
        self.bl_mask = np.zeros((X.shape[0]), dtype=bool)
        self.sms_mask = np.zeros((X.shape[0]), dtype=bool)
        self.tlf_mask = np.zeros((X.shape[0]), dtype=bool)
        self.ott_mask = np.zeros((X.shape[0]), dtype=bool)
        self.norm_mask = np.zeros((X.shape[0]), dtype=bool)
        for i in range(X.shape[0]):
            sample = X[i]
            if sample[self.bl_position]:
                self.bl_mask[i] = True
            #elif sample[self.sms_position]:
            #    self.sms_mask[i] = True
            elif sample[self.tlf_position] == 1:
                self.tlf_mask[i] = True
            elif sample[self.ott_position] == 1:
                self.ott_mask[i] = True
            else:
                self.norm_mask[i] = True

        # Dimensionality reduction.
        self.X = self.tsne.fit_transform(X)

    def plot_data(self):
        if self.X is None:
            raise RuntimeError('There is no data to plot.')

        if self.n_components != 2:
            raise RuntimeError('Cannot plot if n_components is not 2.')

        # Coordinates for all data.
        x_axis_data = self.X[self.norm_mask,0]
        y_axis_data = self.X[self.norm_mask,1]

        # Coordinates for black listed.
        x_axis_black_listed = self.X[self.bl_mask,0]
        y_axis_black_listed = self.X[self.bl_mask,1]

        x_axis_tlf = self.X[self.tlf_mask,0]
        y_axis_tlf = self.X[self.tlf_mask,1]

        x_axis_ott = self.X[self.ott_mask,0]
        y_axis_ott = self.X[self.ott_mask,1]

        print(len(x_axis_data))
        print(len(x_axis_black_listed))
        # Plot.
        plt.scatter(x_axis_data, y_axis_data, color='r', label='Remaining data')
        plt.scatter(x_axis_black_listed, y_axis_black_listed, color='k', label='Black listed')
        plt.scatter(x_axis_tlf, y_axis_tlf, color='g', label='Toll-free')
        plt.scatter(x_axis_ott, y_axis_ott, color='y', label='OTT')

        plt.legend()
        plt.show()
