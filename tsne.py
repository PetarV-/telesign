import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sns.set()

class TSNEReductor():

    def __init__(self, n_components, bl_position=273):
        self.bl_position = bl_position

        self.sms_position = bl_position + 1

        self.mob_position = bl_position - 7
        self.fix_position = bl_position - 6
        self.tch_position = bl_position - 5
        self.ott_position = bl_position - 4
        self.prm_position = bl_position - 3
        self.tlf_position = bl_position - 2

        self.n_components = n_components
        self.tsne = TSNE(n_components)

    def perform(self, X):
        # Remember all important.
        self.bl_mask = np.zeros((X.shape[0]), dtype=bool)

        self.sms_mask = np.zeros((X.shape[0]), dtype=bool)

        self.mob_mask = np.zeros((X.shape[0]), dtype=bool)
        self.fix_mask = np.zeros((X.shape[0]), dtype=bool)
        self.tch_mask = np.zeros((X.shape[0]), dtype=bool)
        self.ott_mask = np.zeros((X.shape[0]), dtype=bool)
        self.prm_mask = np.zeros((X.shape[0]), dtype=bool)
        self.tlf_mask = np.zeros((X.shape[0]), dtype=bool)
        
        self.norm_mask = np.zeros((X.shape[0]), dtype=bool)
        for i in range(X.shape[0]):
            sample = X[i]
            if sample[self.bl_position]:
                self.bl_mask[i] = True
            #elif sample[self.sms_position]:
            #    self.sms_mask[i] = True
            elif sample[self.mob_position] == 1:
                self.mob_mask[i] = True
            elif sample[self.fix_position] == 1:
                self.fix_mask[i] = True
            elif sample[self.tch_position] == 1:
                self.tch_mask[i] = True
            elif sample[self.ott_position] == 1:
                self.ott_mask[i] = True
            elif sample[self.prm_position] == 1:
                self.prm_mask[i] = True
            elif sample[self.tlf_position] == 1:
                self.tlf_mask[i] = True
            else:
                self.norm_mask[i] = True

        # Dimensionality reduction.
        self.X = self.tsne.fit_transform(X)

    def plot_data(self):
        if self.X is None:
            raise RuntimeError('There is no data to plot.')

        if self.n_components != 2:
            raise RuntimeError('Cannot plot if n_components is not 2.')

        # Coordinates for blacklisted.
        x_axis_bl = self.X[self.bl_mask,0]
        y_axis_bl = self.X[self.bl_mask,1]

        # Coordinates for all other important.
        x_axis_mob = self.X[self.mob_mask,0]
        y_axis_mob = self.X[self.mob_mask,1]

        x_axis_fix = self.X[self.fix_mask,0]
        y_axis_fix = self.X[self.fix_mask,1]

        x_axis_tch = self.X[self.tch_mask,0]
        y_axis_tch = self.X[self.tch_mask,1]

        x_axis_ott = self.X[self.ott_mask,0]
        y_axis_ott = self.X[self.ott_mask,1]

        x_axis_prm = self.X[self.prm_mask,0]
        y_axis_prm = self.X[self.prm_mask,1]

        x_axis_tlf = self.X[self.tlf_mask,0]
        y_axis_tlf = self.X[self.tlf_mask,1]

        # Coordinates for remaining data.
        x_axis_norm = self.X[self.norm_mask,0]
        y_axis_norm = self.X[self.norm_mask,1]

        # print(len(x_axis_data))
        # print(len(x_axis_black_listed))

        # Plot.
        plt.scatter(x_axis_data, y_axis_data, color='r', label='Remaining data')

        plt.scatter(x_axis_bl, y_axis_bl, color='k', label='Black listed')

        plt.scatter(x_axis_mob, y_axis_mob, color='c', label='Mobile')
        plt.scatter(x_axis_fix, y_axis_fix, color='m', label='Fixed')
        plt.scatter(x_axis_tch, y_axis_tch, color='#E86712', label='Tech')
        plt.scatter(x_axis_ott, y_axis_ott, color='y', label='OTT')
        plt.scatter(x_axis_prm, y_axis_prm, color='b', label='Premium')
        plt.scatter(x_axis_tlf, y_axis_tlf, color='g', label='Toll-free')

        plt.legend()
        plt.show()
