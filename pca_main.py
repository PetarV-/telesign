import numpy as np
from pca import PCAReductor
import random

n_components = 2
bl_position = 13

X = np.random.rand(900, 1000)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        flag = random.randint(0, 1)
        X[i][j] = flag

pca = PCAReductor(n_components, bl_position)
pca.perform(X)
pca.plot_data()
