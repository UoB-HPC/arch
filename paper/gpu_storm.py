#!/usr/bin/python
import csv
import matplotlib.pyplot as plt
import itertools
import numpy as np
from operator import itemgetter

applications = [ 'Hot', 'Wet', 'Hot n Wet', 'Ideal' ]
labels = [ '2 GPU', '3 GPU', '4 GPU', '5 GPU', '6 GPU', '7 GPU', '8 GPU' ]
results = [[1.466813348, 1.755926251, 2.567394095, 2.808988764, 3.352891869, 3.659652333, 4.509582864],
        [1.382536383, 1.641975309, 2.383512545, 2.57751938, 2.942477876, 3.325, 3.5],
        [0.929809544, 0.88963964, 0.830387004, 0.809878844, 0.758554469, 0.766043724, 0.700693437],
        [2, 3, 4, 5, 6, 7, 8]]

icons = [ 'o-', 'v-', 's-', '>-', 'p-' ]
label_column = 0
header_row = 0
result_columns = [ 1, 2, 3 ]
error_column = 6

def Program():
    font = { 'family' : 'sans-serif', 'weight' : 'bold', 'size' : 12 }
    plt.rc('font', **font)
    fig, ax = plt.subplots(facecolor='white')

    for rr in range(0, len(results)):
        x = np.arange(0, len(labels))
        print x
        print results[rr]
        plt.plot(x, results[rr], icons[rr], label=applications[rr], ms=8.0)

    ind = range(0, len(labels))
    locs, xlabels = plt.xticks(ind, labels, fontsize=11, rotation=90)
    plt.setp(xlabels, rotation=45)
    #ax.set_xlim([1, len(labels)-1])
    #ax.set_ylim([0, 100])

    handles,axlabels=ax.get_legend_handles_labels()
    ax.legend(handles, applications, loc='upper center', 
            bbox_to_anchor=(0.5, 1.25), 
            ncol=4, fancybox=True, shadow=False, prop={'size':12})

    ax.grid(zorder=0)
    plt.title('Performance of Packages Independently and Coupled on Broadwell 44 Core')
    plt.ylabel('Speedup (x)', fontsize=12)
    plt.show()

Program()
