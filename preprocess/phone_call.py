# A data class for phone calls
class PhoneCall():
    allowed_keys = [
        'hash', # string
        'id_a', # int, id ko je pozvao uvek
        'a_unknown', # int 0/1
        'id_b', # int, id ko je pozvan uvek
        'b_unknown', # int 0.1
        'datetime', # datetime
        'orig_op_country', # one-hot vec drzave (isti!)
        'transm_op_country', # one-hot vec drzave (isti!)
        'recv_op_country', # one-hot vec drzave (isti!)
        'dest_op_country', # one-hot vec drzave (isti!)
        'tocs', # one-hot vec duzine 5? (FMNO, wholesale, EasyConnect, DSP, Unknown)
        'roaming', # int 0/1 da li je roaming: FMNO && orig_op != transm_op
        'call_duration', # int (sekunde)
        'setup_duration', # int (sekunde)
        'answered', # int 0/1
        'status_cat', # one-hot vec duzine 3 (network error, sub action, no error)
        'status_name', # one-hot vec len=51 (cat$name) (status_name_mappings.txt)
        'release_dir' # one-hot vec duzine 3 (A prekinuo, oba, B prekinuo)
    ]

    def __init__(self):
        self.features = dict()

    def __getitem__(self, key):
        return self.features[key]

    def __setitem__(self, key, val):
        if key not in PhoneCall.allowed_keys:
            raise RuntimeError('Key {} not allowed in PhoneCall!'.format(key))
        self.features[key] = val

    def __str__(self):
        return str(self.features)

    def __len__(self):
        return len(self.features)
