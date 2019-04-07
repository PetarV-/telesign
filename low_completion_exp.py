import numpy as np
import csv
from datetime import datetime

from dataset import Dataset
from phone_call import PhoneCall
from phone_number import PhoneNumber
from utils import to_array

import pickle

# Experiment with low completion feature
# Code mostly copied from preprocess.py

ts_format = '%Y-%m-%dT%H:%M:%S.%fZ'
#The following part should create mappings between country codes and country names

country_codes_file = open("additional/country_codes.csv")
countries_csv = csv.reader(country_codes_file, delimiter = ',')

country_to_code = dict()
code_to_country = dict()

line_count = 0
for row in countries_csv:
    line_count += 1
    if line_count == 1:
        continue
    country_name = row[0]
    country_code = row[1]
    country_to_code[country_name] = country_code
    code_to_country[country_code] = country_name

country_to_id = dict()
id_to_country = dict()
unknown_ids = set()

countries_num = 0

for ctr in country_to_code:
    countries_num += 1
    country_to_id[ctr] = countries_num
    id_to_country[countries_num] = ctr

# country_file = open("additional/countries_mapping.txt", "w")
# country_file.write(str(id_to_country))

def one_hot_vec(num_elements, which):
    ret = [0] * num_elements
    ret[which - 1] = 1
    return ret

def hot_one(vec):
    for i in range(len(vec)):
        if vec[i] > 0:
            return i + 1

cur_id = 0
hashes_to_id = dict()
id_to_hashes = dict()

numbers_file = open("phone_numbers.csv", encoding = 'utf-8-sig')

line_count = 0

numbers_csv = csv.DictReader(numbers_file, delimiter = ',')

dataset = Dataset()

number_types = {'MOBILE' : 1, 'FIXED' : 2, 'PREMIUM' : 5, 'TECHNICAL' : 3, 'OTT' : 4, 'TOLLFREE' : 6, '' : 7}
tocses = {'F/MNO' : 1, 'Wholesale' : 2, 'EasyConnect' : 3, 'DSP' : 4, 'Unknown' : 5}
status_cats = {'Network' : 1, 'Subscriber' : 2, 'No Error' : 3}
releases = {'Calling party released the call first.' : 1, 'Both parties released the call at the same time.' : 2, 'Called party released the call first.' : 3}
statuses = {'Subscriber$No user responding':1,'No Error$Normal call clearing':2,'Network$No circuit/channel available':3,'Network$Temporary failure':4,'No Error$Send special information tone':5,'Subscriber$Unallocated or unassigned number':6,'Network$Internetworking, unspecified':7,'Network$No route to destination':8,'No Error$Normal, unspecified':9,'Subscriber$Call rejected':10,'Subscriber$User busy':11,'Network$Message not compatible with call state':12,'Subscriber$Destination out of order':13,'Network$Recovery on timer expiry':14,'Network$Service or option not implemented, unspecified':15,'Subscriber$T.301 expired: � User Alerted, No answer from user':16,'Network$Service or option not available, unspecified':17,'Subscriber$Invalid number format or incomplete address':18,'Subscriber$Number changed to number in diagnostic field.':19,'Network$Switching equipment congestion':20,'Network$Protocol error, unspecified':21,'Subscriber$Subscriber absent':22,'Network$Resource unavailable, unspecified':23,'Network$No route to specified transit network (Transit Network Identity)':24,'Subscriber$Bearer capability not authorized':25,'Network$Network out of order':26,'Network$Call resumed':27,'Network$Incompatible destination':28,'Network$Proprietary diagnostic code (not necessarily bad). Typically used to pass proprietary control or maintenance messages between multiplexers.':29,'No Error$Non-selected user clearing':30,'No Error$Intermediate CDR (call not terminated)':31,'Network$Misdialled trunk prefix':32,'Subscriber$Requested facility not subscribed':33,'Network$Bearer service not implemented':34,'Network$Invalid message, unspecified':35,'Network$Requested circuit channel not available':36,'Network$Precedence call blocked':37,'Subscriber$Reverse charging not allowed':38,'Network$Mandatory information element is missing':39,'Network$Bearer capability not presently available':40,'Network$Prefix 0 dialed but not allowed':41,'Network$Invalid transit network selection (national use)':42,'Subscriber$Incoming calls barred within CUG':43,'Network$Information element nonexistent or not implemented':44,'Network$Invalid information element contents':45,'Network$Destination unattainable':46,'Network$Invalid call reference value':47,'No Error$Call suspended':48,'Network$Parameter non-existent or not implemented � passed on':49,'Subscriber$EKTS facility rejected by network':50,'Network$Requested facility not implemented':51}

good_outcalls = dict()
total_outcalls = dict()

# ima ih 9872
for row in numbers_csv:
    #print(row)
    line_count += 1
    phone_number = PhoneNumber()
    cur_id += 1
    hash = row['PHONE_NUMBER']
    hashes_to_id[hash] = cur_id
    id_to_hashes[cur_id] = hash
    phone_number['hash'] = hash
    phone_number['id'] = cur_id
    country_id = country_to_id[code_to_country[row['OPERATOR_COUNTRY_ISO2']]]
    # DON'T ONE HOT COUNTRIES IN [1, 266]
    # phone_number['country'] = one_hot_vec(countries_num, country_id)
    phone_number['country'] = country_id
    n_type = row['NUMBER_TYPE']
    phone_type = one_hot_vec(7, number_types[n_type])
    phone_number['type'] = phone_type
    phone_number['blacklist'] = 0 if row['BLACK_LIST_FLAG'] == 'FALSE' else 1
    phone_number['a2p'] = int(row['A2P_SMS_FLAG'])
    dataset[cur_id] = phone_number
    phone_number['total_outcalls'] = 0
    phone_number['good_outcalls'] = 0
    phone_number['outcall_ratio'] = -1

print("Phone numbers done")

traffic_file = open("voice_traffic.csv", encoding = 'utf-8-sig')

line_count = 0

traffic_csv = csv.DictReader(traffic_file, delimiter = ',')

num_phone_calls = 3441439


# ima ih 3441439
curr_row = 0
pc = 0
for row in traffic_csv:
    curr_row += 1
    if curr_row % 34414 == 0:
        pc += 1
        print("{}/{} ({}%)".format(curr_row, num_phone_calls, pc))
    # print(row)
    line_count += 1
    phone_call = PhoneCall()
    phone_call['hash'] = row['CALL_ID']
    hash_a = row['A_NUMBER']
    # print(hash_a)
    if hash_a not in hashes_to_id:
        cur_id += 1
        hashes_to_id[hash_a] = cur_id
        id_to_hashes[cur_id] = hash_a
    # print(hashes_to_id[hash_a])
    phone_call['a_unknown'] = 0 if dataset.contains(hashes_to_id[hash_a]) else 1
    phone_call['id_a'] = hashes_to_id[hash_a]
    hash_b = row['B_NUMBER']
    if hash_b not in hashes_to_id:
        cur_id += 1
        hashes_to_id[hash_b] = cur_id
        id_to_hashes[cur_id] = hash_b
    phone_call['b_unknown'] = 0 if dataset.contains(hashes_to_id[hash_b]) else 1
    phone_call['id_b'] = hashes_to_id[hash_b]

    # DON'T ONE HOT COUNTRIES IN [1, 266]
    # one_hot_vec(countries_num, orig/transm/recv/dest_country)
    phone_call['orig_op_country'] = country_to_id[row['ORIG_OPER_CTRY']]
    phone_call['transm_op_country'] = country_to_id[row['TRANSM_OPER_CTRY']]
    phone_call['recv_op_country'] = country_to_id[row['RECV_OPER_CTRY']]
    phone_call['dest_op_country'] = country_to_id[row['DEST_OPER_CTRY']]

    row_tocs = row['TRANSM_OPER_COMMERCIAL_SEGMENT']
    phone_call['tocs'] = one_hot_vec(5, tocses[row_tocs])
    timestamp = row['CALL_DATETIME']
    phone_call['datetime'] = datetime.strptime(timestamp, ts_format)
    phone_call['call_duration'] = int(row['CALL_DURATION'])
    phone_call['setup_duration'] = int(row['CALL_SET_UP_DURATION'])
    phone_call['answered'] = 0 if row['CALL_ANSWERED_IND'] == 'N' else 1
    phone_call['status_cat'] = one_hot_vec(3, status_cats[row['CALL_STATUS_CATEGORY']])
    phone_call['status_name'] = one_hot_vec(51, statuses[row['CALL_STATUS_CATEGORY'] + '$' + row['CALL_STATUS_NAME']])
    phone_call['release_dir'] = one_hot_vec(3, releases[row['RELEASE_DIRECTION_DESCRIPTION']])

    # roaming
    # DON'T ONE HOT COUNTRIES IN [1, 266]
    orig_country = phone_call['orig_op_country']
    if not phone_call['a_unknown']:
        orig_country = dataset[phone_call['id_a']]['country']
    phone_call['roaming'] = 1 if row_tocs == 'F/MNO' and orig_country != phone_call['transm_op_country'] else 0
    
    # skippity skip (REMOVE)
    #both = [hash_a, hash_b]
    #if '27aa6048a8af29e92bca' not in both and 'a2e3d9ecd70fae491b2f' not in both:
    #    continue


    # OVDE PLAY
    caller = phone_call['id_a']
    if dataset.contains(caller):
        dataset[caller]['total_outcalls'] += 1
        if phone_call['answered'] and phone_call['call_duration'] > 0:
            dataset[caller]['good_outcalls'] += 1

    # add to lists
    #if not phone_call['a_unknown']:
    #    dataset[phone_call['id_a']]['ts_out'].append(phone_call)
    #if not phone_call['b_unknown']:
    #    dataset[phone_call['id_b']]['ts_in'].append(phone_call)

# Kada se a:country matchuje sa orig_country:
# matches:  2479449
# total: 2588975
# percentage: 0.9576952268755009

pairs = []
for i in range(1, 9873):
    pairs.append((dataset[i]['good_outcalls'], dataset[i]['total_outcalls']))

pairs_nonzero = []
for g, t in pairs:
    if t != 0:
        pairs_nonzero.append((g, t))

pairs_under50 = []
pairs_under100 = []
pairs20 = []
pairs30 = []
pairs40 = []
for g, t in pairs_nonzero:
    if t < 50:
        pairs_under50.append((g, t))
    if t < 100:
        pairs_under100.append((g, t))
    if t >= 20:
        pairs20.append((g, t))
    if t >= 30:
        pairs30.append((g, t))
    if t >= 40:
        pairs40.append((g, t))


# 20 je dobra vrednost: ispod 0.17 856/1677/9873
# 40:

def to_ratios(neka_lista):
    ratios = []
    for g, t in neka_lista:
        ratios.append(g/t)
    return ratios

import code
code.interact(local=locals())

low_completion = []
for i in range(1, 9873):
	lc = False
	if dataset[i]['total_outcalls'] >= 20:
		g = dataset[i]['good_outcalls']
		t = dataset[i]['total_outcalls']
		if (g/t) <= 0.17:
 			lc = True
 	low_completion.append(lc)
print(len(low_completion))
