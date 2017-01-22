#!/usr/bin/python
import csv
import matplotlib.pyplot as plt
import itertools
import numpy as np
from operator import itemgetter

applications = [ 'Hot 2d (tiles)', 'Fast 2d', 'Hot 2d + Fast 2d', 'Ideal' ]
labels = [ '88', '176', '352', '704', '1408', '2816' ]
results = [
        [1.974736842, 3.941176471, 10.86486486, 62.53333333, 93.8, 117.25],
        [1.480620155, 2.220930233, 3.537037037, 4.775, 5.617647059, 7.958333333],
        [1.897340754, 3.575757576, 7.650872818, 19.41772152, 30.07843137, 34.47191011],
        [2, 4, 8, 16, 32, 64]]

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
    locs, xlabels = plt.xticks(ind, labels, fontsize=12)
    #plt.setp(xlabels, rotation=90)
    #ax.set_xlim([1, len(labels)-1])
    #ax.set_ylim([0, 17])

    handles,axlabels=ax.get_legend_handles_labels()
    ax.legend(handles, applications, loc='upper center', 
            bbox_to_anchor=(0.5, 1.25), 
            ncol=5, fancybox=True, shadow=False, prop={'size':12})

    ax.grid(zorder=0)
    plt.title('Performance of Packages Independently and Composed on Broadwell 44 core')
    plt.ylabel('Speedup (x)', fontsize=13)
    plt.xlabel('Cores', fontsize=13)
    plt.show()

Program()
