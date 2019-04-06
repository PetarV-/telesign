import numpy as np

# total number of phones in the dataset
NB_PHONES = 9872
# total number of countries in the dataset
NB_COUNTRIES = 266 
# number of TOCs
NB_TOCS = 5

class FeatureExtractor():

    def _get_mean(self, key, use_only_successful):
        arr = []
        for phone_call in self.phone_calls:
            if use_only_successful:
                if phone_call['status_cat'][2] == 1:
                    arr.append(phone_call[key])
            else:
                arr.append(phone_call[key])

        if len(arr) == 0:
            return 0.0

        return np.mean(arr)

    def _get_var(self, key, use_only_successful):
        arr = []
        for phone_call in self.phone_calls:
            if use_only_successful:
                if phone_call['status_cat'][2] == 1:
                    arr.append(phone_call[key])
            else:
                arr.append(phone_call[key])

        if len(arr) <= 1:
            return 0.0

        return np.std(arr)

    def _get_max(self, key):
        arr = []
        for phone_call in self.phone_calls:
            arr.append(phone_call[key])

        return np.maximum(arr)

    def get_feature_vec(self, phone_calls, is_a):
        self.phone_calls = phone_calls
        feature_vec = []

        # Countries.
        cty_vec_or = np.zeros(NB_COUNTRIES)
        cty_vec_tr = np.zeros(NB_COUNTRIES)
        cty_vec_rc = np.zeros(NB_COUNTRIES)
        cty_vec_ds = np.zeros(NB_COUNTRIES)

        if len(phone_calls) > 0:
            for phone_call in phone_calls:
                cty_vec_or[phone_call['orig_op_country'] - 1] += 1.0
                cty_vec_tr[phone_call['transm_op_country'] - 1] += 1.0
                cty_vec_rc[phone_call['recv_op_country'] - 1] += 1.0
                cty_vec_ds[phone_call['dest_op_country'] - 1] += 1.0

            cty_vec_or /= np.sum(cty_vec_or)
            cty_vec_tr /= np.sum(cty_vec_tr)
            cty_vec_rc /= np.sum(cty_vec_rc)
            cty_vec_ds /= np.sum(cty_vec_ds)

        feature_vec.extend(cty_vec_or)
        feature_vec.extend(cty_vec_tr)
        feature_vec.extend(cty_vec_rc)
        feature_vec.extend(cty_vec_ds)

        # Type of transmitting operator.
        tocs_vec = np.zeros(NB_TOCS)
        if len(phone_calls) > 0:
            for phone_call in phone_calls:
                tocs_vec += phone_call['tocs']

            tocs_vec /= np.sum(tocs_vec)

        feature_vec.extend(tocs_vec)

        # Call duration and setup duration.
        feature_vec.append(self._get_mean('call_duration', True))
        feature_vec.append(self._get_var('call_duration', True))
        feature_vec.append(self._get_mean('setup_duration', True))
        feature_vec.append(self._get_var('setup_duration', True))

        # Answered calls.
        feature_vec.append(self._get_mean('answered', True))
        # feature_vec.append(_get_var('answered', True))

        # Phone calls num.
        feature_vec.append(len(phone_calls))

        # Unique phone numbers.
        unique_phones = set()
        for phone_call in phone_calls:
            if is_a:
                unique_phones.add(phone_call['id_b'])
            else:
                unique_phones.add(phone_call['id_a'])

        feature_vec.append(len(unique_phones))

        return feature_vec

    def get_adj_vec(self, calls_in, calls_out, my_id):
        adj_vec = np.zeros(NB_PHONES)
        adj_vec[my_id] = 1.0

        for phone_call in calls_in:
            if phone_call['a_unknown'] == 0:
                adj_vec[phone_call['id_a']] = 1.0
        # comment out for directed graph
        for phone_call in calls_out:
            if phone_call['b_unknown'] == 0:
                adj_vec[phone_call['id_b']] = 1.0

        return adj_vec

