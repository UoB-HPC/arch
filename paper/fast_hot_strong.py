#!/usr/bin/python
import csv
import matplotlib.pyplot as plt
import itertools
import numpy as np
from operator import itemgetter

applications = [ 'Hot 2d', 'Fast 2d', 'Hot 2d + Fast 2d', 'Ideal' ]
labels = [ '1 node', '2 nodes', '3 nodes', '4 nodes', '8 nodes', '16 nodes', '32 nodes' ]
results = [
        [1.951604599, 2.920390344, 4.114327062, 12.16256684, 50.0969163, 64.98285714 ],
        [1.914634146, 2.854545455, 3.87654321, 7.359375, 20.93333333, 39.25],
        [ 1.956687292, 2.910785933, 4.091904446, 11.46189591, 43.27368421, 62.92346939 ],
        [2, 3, 4, 8, 16, 32]]

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
        x = np.arange(1, len(labels))
        plt.plot(x, results[rr], icons[rr], label=applications[rr], ms=8.0)

    ind = range(0, len(labels))
    locs, xlabels = plt.xticks(ind, labels, fontsize=12, rotation=90)
    plt.setp(xlabels, rotation=45)
    ax.set_xlim([1, len(labels)-1])
    #ax.set_ylim([0, 100])

    handles,axlabels=ax.get_legend_handles_labels()
    ax.legend(handles, applications, loc='upper center', 
            bbox_to_anchor=(0.5, 1.25), 
            ncol=4, fancybox=True, shadow=False, prop={'size':12})

    ax.grid(zorder=0)
    plt.title('Performance of Packages Independently and Coupled on Haswell 32 Core')
    plt.ylabel('Speedup (x)', fontsize=13)
    plt.show()

Program()
