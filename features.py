import numpy as np

class FeatureExtractor():

	def _get_mean(self, key, use_only_successful):
		arr = []
		for phone_call in phone_call:
			if use_only_successful and phone_call['status_cat'][2]:
				arr.add(phone_call[key])

		return np.mean(arr)

	def _get_var(self, key, use_only_successful):
		arr = []
		for phone_call in phone_call:
			if use_only_successful and phone_call['status_cat'][2]:
				arr.add(phone_call[key])

		return np.std(arr)

	def _get_max(self, key):
		arr = []
		for phone_call in phone_call:
			arr.add(phone_call[key])

		return np.maximum(arr)

	def get_feature_vec(self, phone_calls):
		self.phone_calls = phone_calls
		feature_vec = []
		
		# Countries.
		feature_vec.add(_get_max('orig_op_country'))
		feature_vec.add(_get_max('transm_op_country'))
		feature_vec.add(_get_max('recv_op_country'))
		feature_vec.add(_get_max('dest_op_country'))

		# Type of transmitting operator.
		feature_vec.add(_get_max('tocs'))

		# Call duration and setup duration.
		feature_vec.add(_get_mean('call_duration', True))
		feature_vec.add(_get_var('call_duration', True))
		feature_vec.add(_get_mean('setup_duration', True))
		feature_vec.add(_get_var('setup_duration', True))

		# Answered calls.
		feature_vec.add(_get_mean('answered', True))
		feature_vec.add(_get_var('answered', True))

		# Phone calls num.
		feature_vec.add(len(phone_calls))

		# Called phone numbers.
		#called_phones = set()
		#for 

		return feature_vec