from enum import Enum
import numpy as np
import csv

# ----------------------------------------------------
class Status(Enum):
    Network = 0
    Subscriber = 1
    No_error = 2

status_cats = {'Network' : Status.Network,
               'Subscriber' : Status.Subscriber,
               'No Error' : Status.No_error}
# ----------------------------------------------------
releases = {'Calling party released the call first.' : 1,
            'Both parties released the call at the same time.' : 2,
            'Called party released the call first.' : 3}
# ----------------------------------------------------
net_messg = {'Network$No circuit/channel available':3,
             'Network$Temporary failure':4,
             'Network$Internetworking, unspecified':7,
             'Network$No route to destination':8,
             'Network$Message not compatible with call state':12,
             'Network$Recovery on timer expiry':14,
             'Network$Service or option not implemented, unspecified':15,
             'Network$Service or option not available, unspecified':17,
             'Network$Switching equipment congestion':20,
             'Network$Protocol error, unspecified':21,
             'Network$Resource unavailable, unspecified':23,
             'Network$No route to specified transit network (Transit Network Identity)':24,
             'Network$Network out of order':26,
             'Network$Call resumed':27,
             'Network$Incompatible destination':28,
             'Network$Proprietary diagnostic code (not necessarily bad). Typically used to pass proprietary control or maintenance messages between multiplexers.':29,
             'Network$Misdialled trunk prefix':32,
             'Network$Bearer service not implemented':34,
             'Network$Invalid message, unspecified':35,
             'Network$Requested circuit channel not available':36,
             'Network$Precedence call blocked':37,
             'Network$Mandatory information element is missing':39,
             'Network$Bearer capability not presently available':40,
             'Network$Prefix 0 dialed but not allowed':41,
             'Network$Invalid transit network selection (national use)':42,
             'Network$Information element nonexistent or not implemented':44,
             'Network$Invalid information element contents':45,
             'Network$Destination unattainable':46,
             'Network$Invalid call reference value':47,
             'Network$Parameter non-existent or not implemented � passed on':49,
             'Network$Requested facility not implemented':51}


sub_messg = {'Subscriber$No user responding':1,
             'Subscriber$Unallocated or unassigned number':6,
             'Subscriber$Call rejected':10,
             'Subscriber$User busy':11,
             'Subscriber$Destination out of order':13,
             'Subscriber$T.301 expired: � User Alerted, No answer from user':16,
             'Subscriber$Invalid number format or incomplete address':18,
             'Subscriber$Number changed to number in diagnostic field.':19,
             'Subscriber$Subscriber absent':22,
             'Subscriber$Bearer capability not authorized':25,
             'Subscriber$Requested facility not subscribed':33,
             'Subscriber$Reverse charging not allowed':38,
             'Subscriber$Incoming calls barred within CUG':43,
             'Subscriber$EKTS facility rejected by network':50}

noe_messg = {'No Error$Normal call clearing':2,
             'No Error$Send special information tone':5,
             'No Error$Normal, unspecified':9,
             'No Error$Non-selected user clearing':30,
             'No Error$Intermediate CDR (call not terminated)':31,
             'No Error$Call suspended':48,}
# ----------------------------------------------------
status_num = dict()
for key_status in Status:
    status_num[key_status] = 0

net_messg_num = dict()
for key in net_messg:
    net_messg_num[key] = 0

sub_messg_num = dict()
for key in sub_messg:
    sub_messg_num[key] = 0

noe_messg_num = dict()
for key in noe_messg:
    noe_messg_num[key] = 0

none_num = 0
# ----------------------------------------------------
wfile = open("csi_info.txt","w+")

num_phone_calls = 3441439

traffic_file = open("voice_traffic.csv", encoding = 'utf-8-sig')
traffic_csv = csv.DictReader(traffic_file, delimiter = ',')

print('Started with processing...')
curr_row = 0
pc = 0
for row in traffic_csv:
    # Print info.
    curr_row += 1
    if curr_row % 34414 == 0:
        pc += 1
        print("{}/{} ({}%)".format(curr_row, num_phone_calls, pc))

    # Get info about status.
    row_status = status_cats[row['CALL_STATUS_CATEGORY']]
    row_messg = row['CALL_STATUS_CATEGORY'] + '$' + row['CALL_STATUS_NAME']
    status_num[row_status] += 1

    if row_status == Status.Network:
        net_messg_num[row_messg] += 1
    elif row_status == Status.Subscriber:
        sub_messg_num[row_messg] += 1
    elif row_status == Status.No_error:
        noe_messg_num[row_messg] += 1
    else:
        none_num += 1

print('Done with processing...')

# Print information
for key_status in Status:
    wfile.write('{} : {} --> {:.4f} percent of all traffic.\r\n'.
        format(key_status, status_num[key_status], status_num[key_status]/num_phone_calls))

wfile.write('*** STATUS : NETWORK EROR ***')
for key in net_messg_num:
    wfile.write('{} : {} --> {:.4f} percent of all NET traffic.\r\n'.
        format(net_messg[key], net_messg_num[key], net_messg_num[key]/status_num[Status.Network]))

wfile.write('*** STATUS : SUBSCRIBER EROR ***')
for key in sub_messg_num:
    wfile.write('{} : {} --> {:.4f} percent of all SUB traffic.\r\n'.
        format(sub_messg[key], sub_messg_num[key], sub_messg_num[key]/status_num[Status.Subscriber]))

wfile.write('*** STATUS : NO EROR ***')
for key in noe_messg_num:
    wfile.write('{} : {} --> {:.4f} percent of all NOE traffic.\r\n'.
        format(noe_messg[key], noe_messg_num[key], noe_messg_num[key]/status_num[Status.No_error]))

wfile.close()