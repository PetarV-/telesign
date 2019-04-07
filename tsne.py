import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

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
        self.ezy_position = 1341
        self.dsp_position = 1342 
        self.in_position = 1349
        self.out_position = 2425

        self.n_components = n_components

        # CACHED
        # self.tsne = pickle.load(open('tsne.pkl', 'rb'))

        self.tsne = TSNE(n_components)

        # CACHE
        with open('tsne.pkl', 'wb') as lb:
            pickle.dump(self.tsne, lb)

    # prvo je na sta primeniti TSNE, drugo su originalni podaci (optional)
    def perform(self, X, X_orig=None):
        if X_orig is None:
            X_orig = X
        # Remember all black listed.
        self.bl_mask = np.zeros((X_orig.shape[0]), dtype=bool)

        self.ezy_mask = np.zeros((X_orig.shape[0]), dtype=bool)
        self.dsp_mask = np.zeros((X_orig.shape[0]), dtype=bool)

        self.mob_mask = np.zeros((X_orig.shape[0]), dtype=bool)
        self.fix_mask = np.zeros((X_orig.shape[0]), dtype=bool)
        self.tch_mask = np.zeros((X_orig.shape[0]), dtype=bool)
        self.ott_mask = np.zeros((X_orig.shape[0]), dtype=bool)
        self.prm_mask = np.zeros((X_orig.shape[0]), dtype=bool)
        self.tlf_mask = np.zeros((X_orig.shape[0]), dtype=bool)

        self.inc_mask = np.zeros((X_orig.shape[0]), dtype=bool)
        self.sms_mask = np.zeros((X_orig.shape[0]), dtype=bool)
        self.norm_mask = np.zeros((X_orig.shape[0]), dtype=bool)
        for i in range(X_orig.shape[0]):
            sample = X_orig[i]
            if sample[self.bl_position]:
                self.bl_mask[i] = True
            elif sample[self.ezy_position] > 0.1:
                self.ezy_mask[i] = True
            elif sample[self.dsp_position] > 0.1:
                self.dsp_mask[i] = True
            elif sample[self.out_position] - sample[self.in_position] > 100:
                self.inc_mask[i] = True
            elif sample[self.sms_position]:
                self.sms_mask[i] = True
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
        
        with open('low_completion.pklmask', 'rb') as ff:
            self.low_completion_mask = pickle.load(ff)
        
        #import code 
        #code.interact(local=locals())

        # Dimensionality reduction.
        self.X_tsne = self.tsne.fit_transform(X)

    def plot_data(self):
        if self.X_tsne is None:
            raise RuntimeError('There is no data to plot.')

        if self.n_components != 2:
            raise RuntimeError('Cannot plot if n_components is not 2.')


        # Coordinates for all data.
        # Coordinates for black listed.
        x_axis_bl = self.X_tsne[self.bl_mask,0]
        y_axis_bl = self.X_tsne[self.bl_mask,1]

        x_axis_inc = self.X_tsne[self.inc_mask,0]
        y_axis_inc = self.X_tsne[self.inc_mask,1]

        x_axis_sms = self.X_tsne[self.sms_mask,0]
        y_axis_sms = self.X_tsne[self.sms_mask,1]

        x_axis_mob = self.X_tsne[self.mob_mask,0]
        y_axis_mob = self.X_tsne[self.mob_mask,1]

        x_axis_fix = self.X_tsne[self.fix_mask,0]
        y_axis_fix = self.X_tsne[self.fix_mask,1]

        x_axis_tch = self.X_tsne[self.tch_mask,0]
        y_axis_tch = self.X_tsne[self.tch_mask,1]

        x_axis_tlf = self.X_tsne[self.tlf_mask,0]
        y_axis_tlf = self.X_tsne[self.tlf_mask,1]

        x_axis_ott = self.X_tsne[self.ott_mask,0]
        y_axis_ott = self.X_tsne[self.ott_mask,1]

        x_axis_prm = self.X_tsne[self.prm_mask,0]
        y_axis_prm = self.X_tsne[self.prm_mask,1]

        # Coordinates for remaining data.
        x_axis_norm = self.X_tsne[self.norm_mask,0]
        y_axis_norm = self.X_tsne[self.norm_mask,1]

        x_axis_ezy = self.X_tsne[self.ezy_mask,0]
        y_axis_ezy = self.X_tsne[self.ezy_mask,1]

        x_axis_dsp = self.X_tsne[self.dsp_mask,0]
        y_axis_dsp = self.X_tsne[self.dsp_mask,1]

        # NOVA MASKA

        x_axis_lc = self.X_tsne[self.low_completion_mask,0]
        y_axis_lc = self.X_tsne[self.low_completion_mask,1]

        # print(len(x_axis_data))
        # print(len(x_axis_black_listed))

        # Plot.
        plt.scatter(x_axis_norm, y_axis_norm, color='r', label='Remaining data')
        plt.scatter(x_axis_mob, y_axis_mob, color='c', label='Mobile')
        plt.scatter(x_axis_fix, y_axis_fix, color='m', label='Fixed')
        plt.scatter(x_axis_tch, y_axis_tch, color='#E86712', label='Tech')
        plt.scatter(x_axis_ott, y_axis_ott, color='y', label='OTT')
        plt.scatter(x_axis_prm, y_axis_prm, color='b', label='Premium')
        plt.scatter(x_axis_tlf, y_axis_tlf, color='g', label='Toll-free')

        plt.scatter(x_axis_sms, y_axis_sms, color='#800080', label='SMS Received')
        plt.scatter(x_axis_inc, y_axis_inc, color='#FF00FF', label='High incoming')
        plt.scatter(x_axis_dsp, y_axis_dsp, color='#000080', label='DSP')
        plt.scatter(x_axis_lc, y_axis_lc, color='#7F7F7F', label='Low Completion')
        plt.scatter(x_axis_ezy, y_axis_ezy, color='#2F4F4F', label='EasyConnect')
        plt.scatter(x_axis_bl, y_axis_bl, color='k', label='Black listed')

        plt.legend()
        plt.show()
