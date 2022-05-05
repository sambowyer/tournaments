import numpy as np
from typing import List
import random
from utils import *
from Tournament import Tournament 
from ClassicTournaments import *
from SortingTournaments import *
from MABTournaments import *
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import time


def makeStackedBarChart(xticks : List[str], yvaluesStacks : List[List], ylabelStacks : List[str], title : str, xlabel : str, ylabel : str, filename : str, yRange=None, yErrors=None, sizeInches=[17,10]):
    fig, ax = plt.subplots()
    x = np.arange(len(xticks))

    rects = []

    width=0.35
    rects1 = ax.bar(x-width/2, yvalues1, width, label=ylabel1)
    rects2 = ax.bar(x+width/2, yvalues2, width, label=ylabel2)
    if yErrors is not None:
        ax.errorbar(x-width/2, yvalues1, yerr = yErrors[0],fmt='o',ecolor = 'red',color='yellow')
        ax.errorbar(x+width/2, yvalues2, yerr = yErrors[1],fmt='o',ecolor = 'red',color='yellow')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.legend()
    if yRange is not None:
        ax.set_ylim(yRange)

    
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)


    autolabel(rects1)
    autolabel(rects2)

    # fig.tight_layout()

    fig.set_size_inches(sizeInches[0], sizeInches[1])

    plt.savefig(filename)
    plt.close()