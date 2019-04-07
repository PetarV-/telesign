import pickle

class SemiDataset(Dataset):
    # X: 9872 x ~2444 or 9872 x 100
    # Y: 9872 x 1
    # split: 'all', 'train', 'test', 'unlabeled'
    def __init__(self, use_latent = False, split = 'all'):
        if use_latent:
            with open('ae-norm-newdata.pkl', 'rb') as X_file:
                self.X = pickle.load(X_file)
        else:
            with open('fts1.pkl', 'rb') as X_file:
                self.X = pickle.load(X_file)
        print("X shape: ", self.X.shape)
        self.nb_examples = self.X.shape[0]
        self.nb_features = self.X.shape[1]

        with open('labels.pkl', 'rb') as Y_file:
            self.Y = pickle.load(Y_file)
            self.Y = np.reshape(self.Y, (-1, 1))
            # [0, 1, 2, 3, 4 = unknown]
        
        # SPLIT
        if split == 'all':
            return
        
        if split == 'unlabeled':
        
        if split == 'train' or split == 'test':
        
        raise RuntimeError('unknown split: \'{}\''.format(split))
        
        
    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]