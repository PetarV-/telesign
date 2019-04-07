import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set()
plt.rcdefaults()
#fig, ax = plt.subplots()

status_name = ('Network Error', 'Subscriber Error', 'No Error')
y_pos = np.arange(len(status_name))
status_num = [1191587 // 100000, 623335 // 100000, 1626517 // 100000] 

def f():
    # Plot status
    status_name = ('Network Error', 'Subscriber Error', 'No Error')
    y_pos = np.arange(len(status_name))
    status_num = [1191587 // 100000, 623335 // 100000, 1626517 // 100000] 

    ax.barh(y_pos, status_num, align='center',
            color='red', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(status_name)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Number (x 10^6)')
    ax.set_title('Number per status')

    plt.show()

# Plot messages per status
f, axarr = plt.subplots(nrows=3, sharex=True)

net_num = [335243,126414,427727,183742,18576,13961,55268,1733,1249,2819,2743,11740,9013,571,345,133,63,71,21,80,14,3,9,3,1,19,9,6,7,2,2]
y_pos = np.arange(len(net_num))
axarr[0].barh(y_pos, net_num, align='center')
axarr[0].set_title('Network Error')

sub_num = [209729,268781,28652,51600,16537,4533,24524,10000,7367,1491,97,2,19,3]
y_pos = np.arange(len(sub_num))
axarr[1].barh(y_pos, sub_num, align='center')
axarr[1].set_title('Subscriber Error')

noe_num = [1410451,83376,132157,10,521,2]
y_pos = np.arange(len(noe_num))
axarr[2].barh(y_pos, noe_num, align='center')
axarr[2].set_title('No Error')

plt.show()