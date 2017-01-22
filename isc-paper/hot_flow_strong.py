#!/usr/bin/python
import csv
import matplotlib.pyplot as plt
import itertools
import numpy as np
from operator import itemgetter

applications = [ 'Hot 2d', 'Flow 2d', 'Hot 2d + Flow 2d', 'Ideal' ]
labels = [ '88', '176', '352', '704', '1408', '2816' ]
results = [
        [1.973243205, 3.911781128, 10.77017679, 67.36538462, 98.67605634, 136.038835],
        [1.975481611, 3.903114187, 8.417910448, 23.5, 53.71428571, 86.76923077],
        [1.977048755, 3.916516825, 10.61768802, 58.86872587, 98.36774194, 131.4396552],
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
    #plt.setp(xlabels, rotation=45)
    #ax.set_xlim([1, len(labels)-1])
    #ax.set_ylim([0, 100])

    handles,axlabels=ax.get_legend_handles_labels()
    ax.legend(handles, applications, loc='upper center', 
            bbox_to_anchor=(0.5, 1.25), 
            ncol=4, fancybox=True, shadow=False, prop={'size':12})

    ax.grid(zorder=0)
    plt.title('Performance of Packages Independently and Coupled on Broadwell 44 Core')
    plt.ylabel('Speedup (x)', fontsize=13)
    plt.xlabel('Cores', fontsize=13)
    plt.show()

Program()
