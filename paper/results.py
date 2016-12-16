#!/usr/bin/python
import csv
import matplotlib.pyplot as plt
import itertools
import numpy as np
from operator import itemgetter

applications = [ 'Hot 2d', 'Flow 2d', 'Hot 2d + Flow 2d', 'Ideal' ]
labels = [ '1 node', '2 nodes', '3 nodes', '4 nodes' ]
results = [
[1.935064935, 2.749077491, 3.651960784],
[1.925724638, 2.736164736, 3.615646259],
[1.926010838, 2.736452473, 3.614000782],
[2,3,4]]

icons = [ 'o', 'v', 's', '>', 'p' ]
label_column = 0
header_row = 0
result_columns = [ 1, 2, 3 ]
error_column = 6

def Program():
    font = { 'family' : 'sans-serif', 'weight' : 'bold', 'size' : 12 }
    plt.rc('font', **font)
    fig, ax = plt.subplots(facecolor='white')

    for rr in range(0, len(results)):
        x = np.arange(1, len(labels))
        plt.plot(x, results[rr], label=labels[rr])

    ind = range(0, len(labels))
    locs, xlabels = plt.xticks(ind, labels, fontsize=11, rotation=90)
    plt.setp(xlabels, rotation=45)
    ax.set_xlim([1, len(labels)-1])
    #ax.set_ylim([0, 100])

    handles,axlabels=ax.get_legend_handles_labels()
    ax.legend(handles, applications, loc='upper center', 
            bbox_to_anchor=(0.5, 1.25), 
            ncol=4, fancybox=True, shadow=False, prop={'size':12})

    ax.grid(zorder=0)
    plt.title('Performance of Packages Independently and Coupled on NVIDIA K20X')
    plt.ylabel('Speedup (x)', fontsize=12)
    plt.show()

Program()
