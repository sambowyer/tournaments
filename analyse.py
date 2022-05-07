import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from typing import List, Dict 

def getOptimalMABParams():
    df = pd.read_csv("csvs/MABTuningTestsAveraged.csv")

    df["efficiency"] = df["cosine0"]/df["numMatches"]

    df.sort_values("cosine0", ascending=False, inplace=True)
    print(df[["tournament", "efficiency", "explorationFolds", "patience", "maxLockInProportion", "numMatches", "cosine0", "cosine0STD", "correctPositions"]])

    # df.sort_values("numMatches", inplace=True)
    # print(df[["tournament", "numPlayers", "explorationFolds", "patience", "maxLockInProportion", "numMatches", "cosine0", "cosine0STD", "correctPositions"]])

    UCB = df[df["tournament"] == "UCB"]

    TS = df[df["tournament"] == "TS"]

    EG = df[(df["tournament"] != "UCB")]
    EG = EG[(EG["tournament"] != "TS")]

    for table in (UCB, TS, EG):
        table.sort_values("cosine0", ascending=False, inplace=True)
        print(table[["tournament", "efficiency", "explorationFolds", "patience", "maxLockInProportion", "numMatches", "cosine0", "cosine0STD"]].to_string())
        input()

def makeBarChart(xticks : List[str], yvalues : List, title : str, xlabel : str, ylabel : str, filename : str, yRange=None, yError=None, sizeInches=[17,10]):
    fig, ax = plt.subplots()
    rects1 = ax.bar(xticks, yvalues)
    if yError is not None:
        ax.errorbar(xticks, yvalues, yerr = yError,fmt='o',ecolor = 'red',color='yellow')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yRange is not None:
        ax.set_ylim(yRange)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)

    # fig.tight_layout()
    fig.set_size_inches(sizeInches[0], sizeInches[1])

    plt.savefig(filename)
    plt.close()

def makeDoubledBarChart(xticks : List[str], yvalues1 : List, yvalues2 : List, ylabel1 : str, ylabel2 : str, title : str, xlabel : str, ylabel : str, filename : str, yRange=None, yErrors=None, sizeInches=[17,10]):
    fig, ax = plt.subplots()
    x = np.arange(len(xticks))
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

def makeQuintupledBarChart(xticks : List[str], yvalues : List[List], ylabels : List[str], title : str, xlabel : str, ylabel : str, filename : str, yRange=None, yErrors=None, sizeInches=[17,10]):
    fig, ax = plt.subplots()
    x = np.arange(len(xticks))
    width=0.175
    rects1 = ax.bar(x-4*width/5, yvalues[0], width, label=ylabels[0])
    rects2 = ax.bar(x-2*width/5, yvalues[1], width, label=ylabels[1])
    rects3 = ax.bar(x,           yvalues[2], width, label=ylabels[2])
    rects4 = ax.bar(x+2*width/5, yvalues[3], width, label=ylabels[3])
    rects5 = ax.bar(x-4*width/5, yvalues[4], width, label=ylabels[4])
    
    if yErrors is not None:
        ax.errorbar(x-4*width/5, yvalues[0], yerr = yErrors[0],fmt='o',ecolor = 'red',color='yellow')
        ax.errorbar(x-2*width/5, yvalues[1], yerr = yErrors[1],fmt='o',ecolor = 'red',color='yellow')
        ax.errorbar(x,           yvalues[2], yerr = yErrors[2],fmt='o',ecolor = 'red',color='yellow')
        ax.errorbar(x+2*width/5, yvalues[3], yerr = yErrors[3],fmt='o',ecolor = 'red',color='yellow')
        ax.errorbar(x+4*width/5, yvalues[4], yerr = yErrors[4],fmt='o',ecolor = 'red',color='yellow')
    
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
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)


    # fig.tight_layout()

    fig.set_size_inches(sizeInches[0], sizeInches[1])

    plt.savefig(filename)
    plt.close()


