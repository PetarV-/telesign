import numpy as np
import csv
from datetime import datetime

from dataset import Dataset
from phone_call import PhoneCall
from phone_number import PhoneNumber
from utils import to_array

import pickle

dataset = pickle.load(open('dataset.pkl', 'rb'))
ret, adj = to_array(dataset)

with open('fts1.pkl', 'wb') as pf:
    pickle.dump(ret, pf)
with open('adj1.pkl', 'wb') as pf:
    pickle.dump(adj, pf)
