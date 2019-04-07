import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

sns.set()

# t-Distributed Stochastic Neighbor Embedding
# Dimensionality reduction
# Also colors several significant features
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

        self.tsne = TSNE(n_components)

    # First argument: what to apply tSNE on
    # Second argument: original data (for relevant features)
    def perform(self, X, X_orig=None):
        if X_orig is None:
            X_orig = X

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

        # Create masks

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
        
        # Load low_completion mask from file
        with open('low_completion.pklmask', 'rb') as ff:
            self.low_completion_mask = pickle.load(ff)
        
        # Uncomment the next line to skip performing tSNE and load a pickle
        self.X_tsne = pickle.load(open('tsne_pp50.pkl', 'rb'))

        # Comment the following lines to skip tSNE and load a pickle
        
        # self.X_tsne = self.tsne.fit_transform(X)

        # with open('tsne.pkl', 'wb') as lb:
        #     pickle.dump(self.X_tsne, lb)

    # Plots 2D reduced data with significant colors.
    def plot_data(self):
        if self.X_tsne is None:
            raise RuntimeError('There is no data to plot.')

        if self.n_components != 2:
            raise RuntimeError('Cannot plot if n_components is not 2.')

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

        x_axis_norm = self.X_tsne[self.norm_mask,0]
        y_axis_norm = self.X_tsne[self.norm_mask,1]

        x_axis_ezy = self.X_tsne[self.ezy_mask,0]
        y_axis_ezy = self.X_tsne[self.ezy_mask,1]

        x_axis_dsp = self.X_tsne[self.dsp_mask,0]
        y_axis_dsp = self.X_tsne[self.dsp_mask,1]

        x_axis_lc = self.X_tsne[self.low_completion_mask,0]
        y_axis_lc = self.X_tsne[self.low_completion_mask,1]

        ##################################
        # Piece of code that labels data based on our observations
        # This should definitely be moved to another file as it intereferes
        # with other logic in this file
 
        # Information about possible labels
        labelinfo = [('GOOD', 0, '#006400'), 
                     ('FRAUD', 1, '#e60b42'),
                     ('APP', 2, '#f09a00'),
                     ('CALLCENTRE', 3, '#0099cc'),
                     ('UNKNOWN', 4, '#000000')]

        nb_examples = self.X_tsne.shape[0]
        self.labels = np.full((nb_examples, 1), 4)

        # Follow several rules to label data

        # Rule 1: Toll-free i Tech => GOOD
        idxs = np.reshape((self.tlf_mask == True), -1)
        self.labels[idxs] = 0
        idxs = np.reshape((self.tch_mask == True), -1)
        self.labels[idxs] = 0

        # Rule 2: Black-listed/EasyConnect/Low completion/Premium + bottom 4 clusters (conservative) => FRAUD
        self.fraud_mask = np.zeros((nb_examples), dtype=bool)
        self.fraud_mask = np.logical_or(self.bl_mask, self.fraud_mask)
        self.fraud_mask = np.logical_or(self.ezy_mask, self.fraud_mask)
        self.fraud_mask = np.logical_or(self.low_completion_mask, self.fraud_mask)
        self.fraud_mask = np.logical_or(self.prm_mask, self.fraud_mask)

        def bottom4clusters(x, y):
            if x < -30:
                return y < -45
            elif x < -15:
                return y < -50
            elif x < 0:
                return y < -52
            elif x < 20:
                return y < -30
            else:
                return y < -60

        self.position_mask = np.zeros((nb_examples), dtype=bool)
        for i in range(nb_examples):
            x = self.X_tsne[i,0]
            y = self.X_tsne[i,1]
            self.position_mask[i] = bottom4clusters(x, y)

        # Both must be true
        self.fraud_mask = np.logical_and(self.position_mask, self.fraud_mask)
        idxs = np.reshape((self.fraud_mask == True), -1)
        self.labels[idxs] = 1

        # Rule 3: DSP + cluster plavih levo => APP
        def cluster_blue_left(x, y):
            return x < -70 and y < 20
        
        self.position_mask = np.zeros((nb_examples), dtype=bool)
        for i in range(nb_examples):
            x = self.X_tsne[i,0]
            y = self.X_tsne[i,1]
            self.position_mask[i] = cluster_blue_left(x, y)
        
        self.app_mask = np.logical_and(self.position_mask, self.dsp_mask)
        idxs = np.reshape((self.app_mask == True), -1)
        self.labels[idxs] = 2

        # Rule 4: high incoming + bottom right cluster (conservative) => CALL CENTRE
        def cluster_bottom_right(x, y):
            return x > 30 and y < -30

        self.position_mask = np.zeros((nb_examples), dtype=bool)
        for i in range(nb_examples):
            x = self.X_tsne[i,0]
            y = self.X_tsne[i,1]
            self.position_mask[i] = cluster_bottom_right(x, y)
        
        self.cc_mask = np.logical_and(self.position_mask, self.inc_mask)
        idxs = np.reshape((self.cc_mask == True), -1)
        self.labels[idxs] = 3

        # Rule 5: SMS received + nije u spec klusterima (6) => GOOD
        for i in range(nb_examples):
            x = self.X_tsne[i,0]
            y = self.X_tsne[i,1]
            tmp = not cluster_bottom_right(x, y)
            tmp = tmp and not cluster_blue_left(x, y)
            tmp = tmp and not bottom4clusters(x, y)
            self.position_mask[i] = tmp

        self.another_good_mask = np.logical_and(self.position_mask, self.sms_mask)
        idxs = np.reshape((self.another_good_mask == True), -1)
        self.labels[idxs] = 0

        # Yet another use of this same file, at this point we reeeeally need to split this,
        # but there's no time :(
        # This line here discards all the work done up there and just plots the final predictions
        # loaded from a pickle on the tSNE (beware: indices should match)
        with open('predictions.pkl', 'rb') as ff:
            self.labels = pickle.load(ff)

        for c in [0, 1, 2, 3, 4]:
            mask = np.reshape((self.labels == c), -1)
            xs = self.X_tsne[mask, 0]
            ys = self.X_tsne[mask, 1]
            plt.scatter(xs, ys, color=labelinfo[c][2], label=labelinfo[c][0])

        plt.legend()
        plt.show()
        return 
        ##################################
        # Finally, this code was the original code that marks potentially
        # relevant features on a tSNE plot to help us draw conclusions
        
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
