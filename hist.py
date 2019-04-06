import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

durs = []
sets = []

with open('voice_traffic.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[11] == 'Y':
            #if float(row[9]) > 60 and float(row[9]) < 300:
            durs.append(float(row[9]))
            sets.append(float(row[10]))

    plt.hist(sets, 10)
    plt.show()

