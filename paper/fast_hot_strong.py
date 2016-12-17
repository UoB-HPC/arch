#!/usr/bin/python
import csv
import matplotlib.pyplot as plt
import itertools
import numpy as np
from operator import itemgetter

applications = [ 'Hot 2d + Fast 2d', 'Hot 2d (tiles)', 'Fast 2d', 'Ideal' ]
labels = [ '2 nodes', '4 nodes', '8 nodes', '16 nodes' ]
results = [[1.319375322, 5.482812723, 6.185036203, 9.907216495],
        [2.389456869, 7.546922301, 14.5223301, 15.48447205],
        #[2.194392523, 6.546468401, 11.14556962, 11.66225166],
        [0.988993711, 1.270707071, 2.588477366, 6.418367347],
        [2, 4, 8, 16]]

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
        plt.plot(x, results[rr], icons[rr], label=applications[rr], ms=8.0)

    ind = range(0, len(labels))
    locs, xlabels = plt.xticks(ind, labels, fontsize=12, rotation=90)
    plt.setp(xlabels, rotation=45)
    #ax.set_xlim([1, len(labels)-1])
    ax.set_ylim([0, 17])

    handles,axlabels=ax.get_legend_handles_labels()
    ax.legend(handles, applications, loc='upper center', 
            bbox_to_anchor=(0.5, 1.25), 
            ncol=5, fancybox=True, shadow=False, prop={'size':12})

    ax.grid(zorder=0)
    plt.title('Performance of Packages Independently and Coupled on Haswell 32 Core')
    plt.ylabel('Speedup (x)', fontsize=13)
    plt.show()

Program()
