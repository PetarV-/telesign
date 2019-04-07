import numpy as np
import csv
from datetime import datetime

import pickle

with open('predictions.pkl', 'rb') as pred_file:
    preds = pickle.load(pred_file)

with open('probs.pkl', 'rb') as prob_file:
    probs = pickle.load(prob_file)

print(preds.shape)
print(probs.shape)
print(preds[:10])
print(probs[:10])

names = {
    0: 'good',
    1: 'fraud',
    2: 'application',
    3: 'call_center',
    4: 'unknown'
}

np.set_printoptions(suppress=True)

f = open("add.csv","w+")

f.write('PREDICTED_LABEL,P_GOOD,P_FRAUD,P_APPLICATION,P_CALL_CENTER,P_UNKNOWN\n')
for i in range (9872):
    row = ''
    row += names[preds[i][0]]
    row += ','
    for c in range(5):
        prob = '{:.5f}'.format(probs[i][c])
        row += str(prob)
        if c != 4:
            row += ','
    row += '\n'
    f.write(row)

