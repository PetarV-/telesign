import numpy as np

from phone_number import PhoneNumber
from phone_call import PhoneCall 
from features import FeatureExtractor
from preprocess import one_hot_vec

# total number of phones in the dataset
NB_PHONES = 9873
# total number of features we use
NB_FEATURES = 150
# total number of countries
NB_COUNTRIES = 266

xtract = FeatureExtractor()

def to_array(dataset):
    ret = np.empty((NB_PHONES, NB_FEATURES))
    adj = np.zeros((NB_PHONES, NB_PHONES))
    for i in range(NB_PHONES):
        cur_phone = dataset[i + 1]
        ft_vec = []
        # 1. country
        ft_vec.extend(one_hot_vec(NB_COUNTRIES, cur_phone['country']))
        # 2. type
        ft_vec.extend(cur_phone['type'])
        # 3. blacklist
        ft_vec.append(cur_phone['blacklist'])
        # 4. a2p
        ft_vec.append(cur_phone['a2p'])
        # 5. ts_out
        ft_vec.extend(xtract.get_feature_vec(cur_phone['ts_out'], True))
        # 6. ts_in
        ft_vec.extend(xtract.get_feature_vec(cur_phone['ts_in'], False))

        ret[i] = ft_vec
        adj[i] = xtract.get_adj_vec(cur_phone['ts_in'], cur_phone['ts_out'], i)

    return ret, adj

