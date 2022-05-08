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

def makeBarChart(xticks : List[str], yvalues : List[List], ylabels : List[str], title : str, xlabel : str, ylabel : str, filename : str, yRange=None, yErrors=None, sizeInches=[17,10]):
    fig, ax = plt.subplots()
    x = np.arange(len(xticks))
    numBarsPerX = len(yvalues)
    width=0.95/numBarsPerX
    rects = []
    for i in range(numBarsPerX):
        rects.append(ax.bar(x+(2*i+1-numBarsPerX)*width/2, yvalues[i], width, label=str(ylabels[i])))

    if yErrors is not None:
        for i in range(numBarsPerX):
            ax.errorbar(x+(2*i+1-numBarsPerX)*width/2, yvalues[i], yerr = yErrors[i],fmt='o',ecolor = 'red',color='yellow')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.legend()
    if yRange is not None:
        ax.set_ylim(yRange)

    def autolabel(rects, labelPosition):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / labelPosition, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)

    for r in rects:
        autolabel(r, 2 if yErrors is None else 3)

    # fig.tight_layout()

    fig.set_size_inches(sizeInches[0], sizeInches[1])

    plt.savefig(filename)
    plt.close()

def plotFromDF(df : pd.DataFrame, yColumn : str, xColumn : str, xColumn2 : str, errorColumn=None, yColumnDivisor=None, title="", filename="", yRange=None, xlabel="", ylabel="", sizeInches=[17,10]):

    xValues  = df[xColumn].unique()
    x2Values = df[xColumn2].unique()

    yValues  = []
    yErrors  = []

    for x2 in x2Values:
        yValues.append([])
        yErrors.append([])
        for x in xValues:
            temp = df[df[xColumn] == x]
            temp = temp[temp[xColumn2] == x2]
            y    = temp[yColumn]
            
            if yColumnDivisor is not None:
                y /= temp[yColumnDivisor]
            if len(y) == 0:
                y = pd.Series([np.nan])

            yValues[-1].append(y.values[0])
            if yColumnDivisor is not None:
                if len(temp[errorColumn]) == 0:
                    yErrors[-1].append(0)
                else:
                    yErrors[-1].append(temp[errorColumn].values[0])

    if [] in yErrors:
        yErrors = None

    makeBarChart(xValues, yValues, x2Values, title, xlabel, ylabel, filename, yRange, yErrors, sizeInches)

def plotFromCSV(csv : str, yColumn : str, xColumn : str, xColumn2 : str, errorColumn=None, yColumnDivisor=None, title="", filename="", yRange=None, xlabel="", ylabel="", sizeInches=[17,10]):
    df = pd.read_csv(csv)
    plotFromDF(df, yColumn, xColumn, xColumn2, errorColumn, yColumnDivisor, title, filename, yRange, xlabel, ylabel, sizeInches)

def getCorrectPlacesDF(csv : str, numPlayers, header="correctPositions") -> pd.DataFrame:
    df = pd.read_csv(csv)
    
    df = df[df["numPlayers"] == numPlayers]

    numTests = len(df)

    df = pd.concat([df]*4)

    correctPlacesProps = []
    correctPlacesPropsSTD = []
    positionNumbers = []
    
    for i in range(numTests):
        for j in range(4):
            rowNum = i + j*numTests

            positionNumbers.append(["1st","2nd","3rd",f"{numPlayers}nd" if numPlayers%10==2 else f"{numPlayers}th"][j])

            correctPlacesProps.append(float(df[header].values[rowNum].split("_")[[0,1,2,-1][j]]))
            correctPlacesPropsSTD.append(float(df[header+"STD"].values[rowNum].split("_")[[0,1,2,-1][j]]))

    df["positionNumber"] = positionNumbers
    df["positionAccuracy"] = correctPlacesProps
    df["positionAccuracySTD"] = correctPlacesPropsSTD

    # print(df)
    return df

def plotCorrectPlacesFromCSV(csv : str, numPlayers : int, std=True, header="correctPositions", title="", filename="", yRange=None, xlabel="", ylabel="", sizeInches=[17,10]):
    df = getCorrectPlacesDF(csv, numPlayers, header)
    stdColumn = "positionAccuracySTD" if std else None
    plotFromDF(df, "positionAccuracy", "tournament", "positionNumber", stdColumn, None, title, filename, None, "Tournament", "Proportion Of Time With Correctly Ranked ith-best Player", [17,10])

def plotAllCorrectPlaces(csv : str, numPlayers, tournamentName : str, std=True, header="correctPositions", title="", filename="", yRange=None, xlabel="", ylabel="", sizeInches=[17,10]):
    df = pd.read_csv(csv)
    df = df[df["numPlayers"] == numPlayers]
    df = df[df["tournament"] == tournamentName]

    numTests = len(df)

    df = pd.concat([df]*numPlayers)

    correctPlacesProps = []
    correctPlacesPropsSTD = []
    positionNumbers = []
    
    for i in range(numTests):
        for j in range(numPlayers):
            rowNum = i + j*numTests
            pos = str(j+1)
            if (j+1) % 10 == 1:
                pos += "st"
            elif (j+1) % 10 == 1:
                pos += "nd"
            elif (j+1) % 10 == 1:
                pos += "rd"
            else:
                pos += "th"

            positionNumbers.append(pos)

            correctPlacesProps.append(float(df[header].values[rowNum].split("_")[j]))
            correctPlacesPropsSTD.append(float(df[header+"STD"].values[rowNum].split("_")[j]))

    # df["positionNumber"] = positionNumbers
    # df["positionAccuracy"] = correctPlacesProps

    fig, ax = plt.subplots()

    ax.bar(positionNumbers, correctPlacesProps)

    if std:
        ax.errorbar(positionNumbers, correctPlacesProps, yerr = correctPlacesPropsSTD,fmt='o',ecolor = 'red',color='yellow')


    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yRange is not None:
        ax.set_ylim(yRange)

    # fig.tight_layout()

    fig.set_size_inches(sizeInches[0], sizeInches[1])

    plt.savefig(filename)
    plt.close()

    


# plotFromCSV("csvs/MABMainTestsAveraged.csv", "cosine0", "tournament", "numPlayers", "cosine0STD", "numMatches", "ooh", "ahh.png", None, "Tournament", "Cosine Similarity", [17,10])
# d = getCorrectPlacesDF("csvs/MABMainTestsAveraged.csv", 8)
# plotFromDF(d, "positionAccuracy", "tournament", "positionNumber", None, None, "ooh2", "ahh2.png", None, "Tournament", "Proportion Correct Similarity", [17,10])
plotCorrectPlacesFromCSV("csvs/MABMainTestsAveraged.csv", 8, True, title="ooh2", filename="ahh2.png")
plotAllCorrectPlaces("csvs/classicalAndSortingTestsAveraged.csv", 16, "RR100", True, title="ooh3", filename="ahh3.png", xlabel="Position", ylabel="Proportion Of Time With Correctly Ranked ith-best Player")
