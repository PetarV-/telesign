class PhoneNumber():

    # top level je samo array of phone numbers
    allowed_keys = [
        'hash', # str
        'id', # int
        'country', # one-hot vec duzine 200+
        'type', # one-hot vec duzine 7: mob, fix, tech, ott, prem, tollfree, unknown
        'blacklist', # int 0/1
        'a2p', # int 0/1
        'ts_out', # array of PhoneCall when I called, sorted by datetime
        'ts_in' # array of PhoneCall when I was called, sorted by datetime
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