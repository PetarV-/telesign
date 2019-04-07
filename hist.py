import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

sns.set()

durs = []
sets = []

with open('phone_numbers.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    tot = 0
    for _ in reader:
        tot += 1
print(tot)
exit(0)

with open('voice_traffic.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[11] == 'Y':
            #if float(row[9]) > 60 and float(row[9]) < 300:
            durs.append(float(row[9]))
            sets.append(float(row[10]))

    plt.hist(sets, 10)
    plt.show()

