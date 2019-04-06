import numpy as np
import csv
from datetime import datetime
from dataset import Dataset
from phone_call import PhoneCall
from phone_number import PhoneNumber

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

numbers_file = open("task_and_samples/phone_numbers.csv", encoding = 'utf-8-sig')

line_count = 0

numbers_csv = csv.DictReader(numbers_file, delimiter = ',')

dataset = Dataset()

number_types = {'MOBILE' : 1, 'FIXED' : 2, 'PREMIUM' : 5, 'TECHNICAL' : 3, 'OTT' : 4, 'TOLLFREE' : 6, '' : 7}
tocses = {'F/MNO' : 1, 'Wholesale' : 2, 'EasyConnect' : 3, 'DSP' : 4, 'Unknown' : 5}
status_cats = {'Network' : 1, 'Subscriber' : 2, 'No Error' : 3}
releases = {'Calling party released the call first.' : 1, 'Both parties released the call at the same time.' : 2, 'Called party released the call first.' : 3}
statuses = {'Subscriber$No user responding':1,'No Error$Normal call clearing':2,'Network$No circuit/channel available':3,'Network$Temporary failure':4,'No Error$Send special information tone':5,'Subscriber$Unallocated or unassigned number':6,'Network$Internetworking, unspecified':7,'Network$No route to destination':8,'No Error$Normal, unspecified':9,'Subscriber$Call rejected':10,'Subscriber$User busy':11,'Network$Message not compatible with call state':12,'Subscriber$Destination out of order':13,'Network$Recovery on timer expiry':14,'Network$Service or option not implemented, unspecified':15,'Subscriber$T.301 expired: � User Alerted, No answer from user':16,'Network$Service or option not available, unspecified':17,'Subscriber$Invalid number format or incomplete address':18,'Subscriber$Number changed to number in diagnostic field.':19,'Network$Switching equipment congestion':20,'Network$Protocol error, unspecified':21,'Subscriber$Subscriber absent':22,'Network$Resource unavailable, unspecified':23,'Network$No route to specified transit network (Transit Network Identity)':24,'Subscriber$Bearer capability not authorized':25,'Network$Network out of order':26,'Network$Call resumed':27,'Network$Incompatible destination':28,'Network$Proprietary diagnostic code (not necessarily bad). Typically used to pass proprietary control or maintenance messages between multiplexers.':29,'No Error$Non-selected user clearing':30,'No Error$Intermediate CDR (call not terminated)':31,'Network$Misdialled trunk prefix':32,'Subscriber$Requested facility not subscribed':33,'Network$Bearer service not implemented':34,'Network$Invalid message, unspecified':35,'Network$Requested circuit channel not available':36,'Network$Precedence call blocked':37,'Subscriber$Reverse charging not allowed':38,'Network$Mandatory information element is missing':39,'Network$Bearer capability not presently available':40,'Network$Prefix 0 dialed but not allowed':41,'Network$Invalid transit network selection (national use)':42,'Subscriber$Incoming calls barred within CUG':43,'Network$Information element nonexistent or not implemented':44,'Network$Invalid information element contents':45,'Network$Destination unattainable':46,'Network$Invalid call reference value':47,'No Error$Call suspended':48,'Network$Parameter non-existent or not implemented � passed on':49,'Subscriber$EKTS facility rejected by network':50,'Network$Requested facility not implemented':51}

for row in numbers_csv:
    print(row)
    line_count += 1
    phone_call = PhoneNumber()
    cur_id += 1
    hash = row['PHONE_NUMBER']
    hashes_to_id[hash] = cur_id
    id_to_hashes[cur_id] = hash
    phone_call['hash'] = hash
    phone_call['id'] = cur_id
    phone_call['country'] = one_hot_vec(countries_num, country_to_id[code_to_country[row['OPERATOR_COUNTRY_ISO2']]])
    n_type = row['NUMBER_TYPE']
    phone_type = one_hot_vec(7, number_types[n_type])
    phone_call['blacklist'] = 0 if row['BLACK_LIST_FLAG'] == 'FALSE' else 1
    phone_call['a2p'] = int(row['A2P_SMS_FLAG'])
    dataset[cur_id] = phone_call

traffic_file = open("task_and_samples/voice_traffic.csv", encoding = 'utf-8-sig')

line_count = 0

traffic_csv = csv.DictReader(traffic_file, delimiter = ',')

matches = 0
mismatches = 0
for row in traffic_csv:
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
    orig_country = country_to_id[row['ORIG_OPER_CTRY']]
    phone_call['orig_op_country'] = one_hot_vec(countries_num, orig_country)
    if not phone_call['a_unknown']:
        if orig_country == hot_one(dataset[phone_call['id_a']]['country']):
            matches += 1
        else:
            mismatches += 1
    phone_call['transm_op_country'] = one_hot_vec(countries_num, country_to_id[row['TRANSM_OPER_CTRY']])
    phone_call['recv_op_country'] = one_hot_vec(countries_num, country_to_id[row['RECV_OPER_CTRY']])
    phone_call['dest_op_country'] = one_hot_vec(countries_num, country_to_id[row['DEST_OPER_CTRY']])
    row_tocs = row['TRANSM_OPER_COMMERCIAL_SEGMENT']
    phone_call['tocs'] = one_hot_vec(5, tocses[row_tocs])
    timestamp = row['CALL_DATETIME']
    phone_call['datetime'] = datetime.strptime(timestamp, ts_format)
    phone_call['call_duration'] = int(row['CALL_DURATION'])
    phone_call['setup_duration'] = int(row['CALL_SET_UP_DURATION'])
    phone_call['answered'] = 0 if row['CALL_ANSWERED_IND'] == 'N' else 1
    phone_call['status_cat'] = one_hot_vec(3, status_cats[row['CALL_STATUS_CATEGORY']])
    phone_call['status_name'] = one_hot_vec(51, statuses[row['CALL_STATUS_CATEGORY'] + '$' + row['CALL_STATUS_NAME']])
    phone_call['roaming'] = 1 if row_tocs == 'F/MNO' and phone_call['orig_op_country'] != phone_call['transm_op_country'] else 0
    phone_call['release_dir'] = one_hot_vec(3, releases[row['RELEASE_DIRECTION_DESCRIPTION']])
    if not phone_call['a_unknown']:
        dataset[phone_call['id_a']]['ts_out'].append(phone_call)
    if not phone_call['b_unknown']:
        dataset[phone_call['id_b']]['ts_in'].append(phone_call)

print('matches: ' + matches)
print('total: ' + matches + mismatches)

print('percentage: ' + matches / (matches + mismatches))

for p in dataset.values():
    p['ts_out'] = sorted(p['ts_out'], key = lambda x : x['datetime'])
    p['ts_in'] = sorted(p['ts_in'], key = lambda x : x['datetime'])

import code

code.interact(local = locals())
