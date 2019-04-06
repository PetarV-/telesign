class PhoneCall():
    allowed_keys = [
        'id', # int
        'id_called', # int, id onoga koga smo pozvali
        'datetime', # datetime
        'orig_op_country', # one-hot vec drzave (isti!)
        'transm_op_country', # one-hot vec drzave (isti!)
        'recv_op_country', # one-hot vec drzave (isti!)
        'dest_op_country', # one-hot vec drzave (isti!)
        'tocs', # one-hot vec duzine 4? (FMNO, wholesale, EasyConnect, DSP)
        'roaming', # int 0/1 da li je roaming: FMNO && orig_op != transm_op
        'call_duration', # int (sekunde)
        'setup_duration', # int (sekunde)
        'answered', # int 0/1
        'status_cat', # one-hot vec duzine 3? (network error, sub action, no error)
        'status_name', # string??? videti koji su pa mozda one-hot !!!
        'release_dir' # string??? moze realno da bude one-hot !!!
    ]

    def __init__(self):
        self.features = dict()
    
    def __getitem__(self, key):
        return self.features[key]
    
    def __setitem__(self, key, val):
        if key not in allowed_keys:
            raise RuntimeError('Key {} not allowed in PhoneNumber!'.format(key))
        self.features[key] = val
    
    def __str__(self):
        return str(self.features)
    
    def __len__(self):
        return len(self.features)