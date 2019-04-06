from autoencoder import AutoencoderDriver
import numpy as np

nb_examples = 9872
nb_features = 2427

X = np.random.rand(nb_examples, nb_features)

ad = AutoencoderDriver(nb_features, 100, batch_size=128, nb_epochs=10)
ad.run(X)