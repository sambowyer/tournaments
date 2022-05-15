import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from typing import List, Dict 
from tabulate import tabulate

gridLines = True

def getOptimalMABParams(latex=False):
    df = pd.read_csv("csvs/MABTuningTestsAveraged.csv")

    df["efficiency"] = df["cosine1"]/df["numMatches"]

    df.sort_values("cosine1", ascending=False, inplace=True)
    print(df[["tournament", "efficiency", "explorationFolds", "patience", "maxLockInProportion", "numMatches", "cosine1", "cosine1STD", "correctPositions"]])

    # df.sort_values("numMatches", inplace=True)
    # print(df[["tournament", "numPlayers", "explorationFolds", "patience", "maxLockInProportion", "numMatches", "cosine0", "cosine0STD", "correctPositions"]])

    UCB = df[df["tournament"] == "UCB"]

    TS = df[df["tournament"] == "TS"]

    EG = df[(df["tournament"] != "UCB")]
    EG = EG[(EG["tournament"] != "TS")]

    for table in (UCB, TS, EG):
        for n in [8,32]:
            table2 = table[table["numPlayers"] == n]
            table2.sort_values("cosine1", ascending=False, inplace=True)
            # outputTable = table2[["tournament", "numPlayers", "explorationFolds", "patience", "maxLockInProportion", "numMatches", "cosine1", "cosine1STD", "efficiency"]]
            # outputTable = table2[["tournament", "explorationFolds", "patience", "maxLockInProportion", "numMatches", "cosine1"]]
            outputTable = table2[["explorationFolds", "patience", "maxLockInProportion", "numMatches", "cosine1"]]

            if not latex:
                print(outputTable.to_string())
            else:
                # print(tabulate(outputTable[:10], tablefmt="latex", headers=outputTable.columns, showindex=False))
                print(tabulate(outputTable[:10], tablefmt="latex", headers=["Exploration Folds", "Patience", "Max Lock-In Prop.", "Matches", "Cosine Sim."], showindex=False))

def makeBarChart(xticks : List[str], yvalues : List[List], ylabels : List[str], title : str, xlabel : str, ylabel : str, filename : str, yRange=None, yErrors=None, sizeInches=[17,10], legendTitle=None, legendLoc="lower right", yLimScaleFactors=[0.99, 1.02]):
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

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    # ax.yticks(fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)#, fontsize=14)
    # ax.set_xticklabels(x, labelsize=14)
    # ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='both', labelsize=14)

    legend = ax.legend(loc=legendLoc, fontsize=14, title_fontsize=14)
    if legendTitle is not None:
        legend.set_title(legendTitle)

    if yRange is not None:
        ax.set_ylim(yRange)
    else:
        # currentYLim = ax.get_ylim()
        # ax.set_ylim([currentYLim[0], 1.05*currentYLim[1]])
        if yLimScaleFactors is not None:
            ax.set_ylim(yLimScaleFactors[0]*min([min(ys) for ys in yvalues]), yLimScaleFactors[1]*max([max(ys) for ys in yvalues]))

        # ax.relim()
        # ax.autoscale_view()

    def autolabel(rects, labelPosition):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            # ax.annotate('{:.3}'.format(height),
            ax.annotate('{:3G}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / labelPosition, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, size=14)

    if gridLines:
        plt.grid(axis = 'y')
    else:
        for r in rects:
            autolabel(r, 2 if yErrors is None else 3.3)

    # fig.tight_layout()

    fig.set_size_inches(sizeInches[0], sizeInches[1])

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plotFromDF(df : pd.DataFrame, yColumn : str, xColumn : str, xColumn2 : str, errorColumn=None, yColumnDivisor=None, title="", filename="", yRange=None, xlabel="", ylabel="", sizeInches=[17,10], legendTitle=None, legendLoc="upper right", yLimScaleFactors=[0.99, 1.02]):

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
            if errorColumn is not None:
                if len(temp[errorColumn]) == 0:
                    yErrors[-1].append(0)
                else:
                    yErrors[-1].append(temp[errorColumn].values[0])

    if [] in yErrors:
        yErrors = None

    makeBarChart(xValues, yValues, x2Values, title, xlabel, ylabel, filename, yRange, yErrors, sizeInches, legendTitle, legendLoc, yLimScaleFactors)

def plotFromCSV(csv : str, yColumn : str, xColumn : str, xColumn2 : str, errorColumn=None, yColumnDivisor=None, title="", filename="", yRange=None, xlabel="", ylabel="", sizeInches=[17,10]):
    df = pd.read_csv(csv)
    plotFromDF(df, yColumn, xColumn, xColumn2, errorColumn, yColumnDivisor, title, filename, yRange, xlabel, ylabel, sizeInches)

def getCorrectPlacesDF(df : pd.DataFrame, numPlayers, header="correctPositions") -> pd.DataFrame:    
    df = df[df["numPlayers"] == numPlayers]

    numTests = len(df)

    df = pd.concat([df]*4)

    correctPlacesProps = []
    correctPlacesPropsSTD = []
    positionNumbers = []
    
    for j in range(4):
        for i in range(numTests):
            rowNum = i + j*numTests

            positionNumbers.append(["1st","2nd","3rd",f"{numPlayers}nd" if numPlayers%10==2 else f"{numPlayers}th"][j])

            correctPlacesProps.append(float(df[header].values[rowNum].split("_")[[0,1,2,-1][j]]))
            correctPlacesPropsSTD.append(float(df[header+"STD"].values[rowNum].split("_")[[0,1,2,-1][j]]))

    df["positionNumber"] = positionNumbers
    df["positionAccuracy"] = correctPlacesProps
    df["positionAccuracySTD"] = correctPlacesPropsSTD

    # print(df)
    return df

def getCorrectPlacesDFFromCSV(csv : str, numPlayers, header="correctPositions") -> pd.DataFrame:
    return getCorrectPlacesDF(pd.read_csv(csv), numPlayers, header)

def plotCorrectPlacesFromCSV(csv : str, numPlayers : int, std=True, header="correctPositions", title="", filename="", yRange=None, xlabel="", ylabel="", sizeInches=[17,10]):
    df = getCorrectPlacesDFFromCSV(csv, numPlayers, header)
    stdColumn = "positionAccuracySTD" if std else None
    plotFromDF(df, "positionAccuracy", "tournament", "positionNumber", stdColumn, None, title, filename, [0,1], "Tournament", "Proportion Of Tests With Correctly Ranked ith-Best Player", [17,10], "Position")

def plotCorrectPlacesFromDF(df : pd.DataFrame, numPlayers : int, std=True, header="correctPositions", title="", filename="", yRange=None, xlabel="", ylabel="", sizeInches=[17,10]):
    df = getCorrectPlacesDF(df, numPlayers, header)
    stdColumn = "positionAccuracySTD" if std else None
    plotFromDF(df, "positionAccuracy", "tournament", "positionNumber", stdColumn, None, title, filename, [0,1], "Tournament", "Proportion Of Tests With Correctly Ranked ith-Best Player", [17,10], "Position")


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


    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    # ax.yticks(fontsize=16)
    # ax.xticks(fontsize=16)
    # ax.tick_params(axis='x', labelsize=14)
    if yRange is not None:
        ax.set_ylim(yRange)

    # fig.tight_layout()

    fig.set_size_inches(sizeInches[0], sizeInches[1])

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# plotFromCSV("csvs/MABMainTestsAveraged.csv", "cosine0", "tournament", "numPlayers", "cosine0STD", "numMatches", "ooh", "ahh.png", None, "Tournament", "Cosine Similarity", [17,10])
# d = getCorrectPlacesDF("csvs/MABMainTestsAveraged.csv", 8)
# plotFromDF(d, "positionAccuracy", "tournament", "positionNumber", None, None, "ooh2", "ahh2.png", None, "Tournament", "Proportion Correct Similarity", [17,10])
# plotCorrectPlacesFromCSV("csvs/MABMainTestsAveraged.csv", 8, True, title="ooh2", filename="ahh2.png")
for t in ["RR1", "SE", "SE3pp", "DE", "SW", "IS1", "BIS1", "BS1", "SS1", "QS1", "MS1", "HS1"]:
    plotAllCorrectPlaces("csvs/classicalAndSortingTestsAveraged.csv", 16, t, False, title="Proportion Of With Correctly Ranked ith-Best Player", filename=f"img/report_images/props/{t}Props.png", xlabel="Position", ylabel="Proportion")
    plotAllCorrectPlaces("csvs/classicalAndSortingTestsAveraged.csv", 16, t, True, title="Proportion Of With Correctly Ranked ith-Best Player", filename=f"img/report_images/props/{t}PropsSTD.png", xlabel="Position", ylabel="Proportion")


def filterDF(df : pd.DataFrame, column : str, validValues : List) -> pd.DataFrame:
    dfs = []
    for val in validValues:
        dfs.append(df[df[column] == val])
    return pd.concat(dfs)
    
def makeAccuracyPlots(df : pd.DataFrame, xColumn2 : str, filenamePrefix : str, meanNumMatches=True, legendTitle=None, xColumn="tournament", legLocs=["upper right"]*4, xlabel="Tournament"):
    mean = "Mean " if meanNumMatches else ""

    plotFromDF(df, "cosine0", xColumn, xColumn2, None, None, title="Cosine Similarity To True Ranking", filename=filenamePrefix+"cosine0.png", yRange=None, xlabel=xlabel, ylabel="Cosine Similarity", legendTitle=legendTitle, legendLoc=legLocs[0])
    plotFromDF(df, "cosine0", xColumn, xColumn2, "cosine0STD", None, title="Cosine Similarity To True Ranking", filename=filenamePrefix+"cosine0wErr.png", yRange=None, xlabel=xlabel, ylabel="Cosine Similarity", legendTitle=legendTitle, legendLoc=legLocs[0])
    plotFromDF(df, "cosine0", xColumn, xColumn2, None, "numMatches", title=f"Cosine Similarity To True Ranking Divided By {mean}Number Of Matches In The Tournament", filename=filenamePrefix+"cosine0byNumMatches.png", yRange=None, xlabel=xlabel, ylabel="Cosine Similarity / Number of Matches", legendTitle=legendTitle, legendLoc=legLocs[1], yLimScaleFactors=[0.9, 1.125])
    plotFromDF(df, "cosine0", xColumn, xColumn2, None, "numRounds", title=f"Cosine Similarity To True Ranking Divided By {mean}Number Of Rounds In The Tournament", filename=filenamePrefix+"cosine0byNumRounds.png", yRange=None, xlabel=xlabel, ylabel="Cosine Similarity / Number of Rounds", legendTitle=legendTitle, legendLoc=legLocs[1], yLimScaleFactors=[0.9, 1.125])

    plotFromDF(df, "cosine1", xColumn, xColumn2, None, None, title="Cosine Similarity To True Ranking", filename=filenamePrefix+"cosine1.png", yRange=None, xlabel=xlabel, ylabel="Cosine Similarity", legendTitle=legendTitle, legendLoc=legLocs[0])
    plotFromDF(df, "cosine1", xColumn, xColumn2, "cosine1STD", None, title="Cosine Similarity To True Ranking", filename=filenamePrefix+"cosine1wErr.png", yRange=None, xlabel=xlabel, ylabel="Cosine Similarity", legendTitle=legendTitle, legendLoc=legLocs[0])
    plotFromDF(df, "cosine1", xColumn, xColumn2, None, "numMatches", title=f"Cosine Similarity To True Ranking Divided By {mean}Number Of Matches In The Tournament", filename=filenamePrefix+"cosine1byNumMatches.png", yRange=None, xlabel=xlabel, ylabel="Cosine Similarity / Number of Matches",legendTitle=legendTitle, legendLoc=legLocs[1], yLimScaleFactors=[0.9, 1.125])
    plotFromDF(df, "cosine1", xColumn, xColumn2, None, "numRounds", title=f"Cosine Similarity To True Ranking Divided By {mean}Number Of Rounds In The Tournament", filename=filenamePrefix+"cosine1byNumRounds.png", yRange=None, xlabel=xlabel, ylabel="Cosine Similarity / Number of Rounds", legendTitle=legendTitle, legendLoc=legLocs[1], yLimScaleFactors=[0.9, 1.125])

    plotFromDF(df, "cosine0STD", xColumn, xColumn2, None, None, title="Standard Deviation Of Cosine Similarity To True Ranking", filename=filenamePrefix+"cosine0STD.png", yRange=None, xlabel=xlabel, ylabel="Standard Deviation of Cosine Similarity", legendTitle=legendTitle, legendLoc=legLocs[2], yLimScaleFactors=[0.9, 1.125])
    plotFromDF(df, "cosine1STD", xColumn, xColumn2, None, None, title="Standard Deviation Of Cosine Similarity To True Ranking", filename=filenamePrefix+"cosine1STD.png", yRange=None, xlabel=xlabel, ylabel="Standard Deviation of Cosine Similarity", legendTitle=legendTitle, legendLoc=legLocs[2], yLimScaleFactors=[0.9, 1.125])

    plotFromDF(df, "numMatches", xColumn, xColumn2, None, None, title=f"{mean}Number of Matches In Tournament", filename=filenamePrefix+"matches.png", yRange=None, xlabel=xlabel, ylabel="Number Of Matches", legendTitle=legendTitle, legendLoc=legLocs[3], yLimScaleFactors=[0.9, 1.125])
    plotFromDF(df, "numRounds", xColumn, xColumn2, None, None, title=f"{mean}Number of Rounds In Tournament", filename=filenamePrefix+"rounds.png", yRange=None, xlabel=xlabel, ylabel="Number Of Rounds", legendTitle=legendTitle, legendLoc=legLocs[3], yLimScaleFactors=[0.9, 1.125])

# def fixRoundNums(df : pd.DataFrame) -> pd.DataFrame:
#     '''Need to fix the round number calculations and the easiest way to do that seems to be to fix them all with one function.'''
#     # First we fix DE
#     def correctRoundsDE(rounds : int, matches : )

getOptimalMABParams()
# getOptimalMABParams(True)
# input("Generate Graphs?")

# Now actually create the images (once with gridlines and once with annotations instead)

for rootFolder in ["img/report_images/", "img/report_images/detailed/"]:
    # First the basic performance tests from both classical and sorting tournaments
    classicalAndSortingDF = pd.read_csv("csvs/classicalAndSortingTestsAveraged.csv")

    classicalNames = [f"RR{x}" for x in [1,2,3,5,10,25,50,100]] + ["SE", "SE3pp", "DE", "SW"]
    sortingAlgos   = ["IS", "BIS", "BS", "SS", "QS", "MS", "HS"]
    bestOf1SortingNames = [algo+"1" for algo in sortingAlgos]
    sortingNames = bestOf1SortingNames.copy()
    for bestOf in [1,3,5,7,9,51,101]:
        sortingNames += [algo+str(bestOf) for algo in sortingAlgos]

    classicalDF = filterDF(classicalAndSortingDF, "tournament", classicalNames)
    sortingDF   = filterDF(classicalAndSortingDF, "tournament", bestOf1SortingNames)
    sortingDF["tournament"] = [name[:2] if name[:3] != "BIS" else "BIS" for name in sortingDF["tournament"].values]

    makeAccuracyPlots(classicalDF, "numPlayers", f"{rootFolder}classical/", False, "Number of Players", legLocs=["upper right", "upper left", "upper center", "upper right"])
    makeAccuracyPlots(sortingDF, "numPlayers", f"{rootFolder}sorting/", True, "Number of Players", legLocs=["lower right", "upper right", "upper right", "upper right"])

    for numPlayers in classicalDF["numPlayers"].unique():
        plotCorrectPlacesFromDF(classicalDF, numPlayers, False, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}classical/correctPlaces{numPlayers}.png", yRange=None, xlabel="Tournament", ylabel="Proportion")
        plotCorrectPlacesFromDF(classicalDF, numPlayers, True, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}classical/correctPlaces{numPlayers}wErr.png", yRange=None, xlabel="Tournament", ylabel="Proportion")

    for numPlayers in sortingDF["numPlayers"].unique():
        plotCorrectPlacesFromDF(sortingDF, numPlayers, False, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}sorting/correctPlaces{numPlayers}.png", yRange=None, xlabel="Tournament", ylabel="Proportion")
        plotCorrectPlacesFromDF(sortingDF, numPlayers, True, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}sorting/correctPlaces{numPlayers}wErr.png", yRange=None, xlabel="Tournament", ylabel="Proportion")


    classicalDFClearer = filterDF(classicalAndSortingDF, "tournament", [f"RR{x}" for x in [1]] + ["SE", "SE3pp", "DE", "SW"])
    classicalDFClearer = filterDF(classicalDFClearer, "numPlayers", [4,8,16,32])
    makeAccuracyPlots(classicalDFClearer, "numPlayers", f"{rootFolder}classical/clearer/", False, "Number of Players", legLocs=["upper right", "upper left", "upper center", "upper right"])
    for numPlayers in classicalDFClearer["numPlayers"].unique():
        plotCorrectPlacesFromDF(classicalDFClearer, numPlayers, False, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}classical/clearer/correctPlaces{numPlayers}.png", yRange=None, xlabel="Tournament", ylabel="Proportion")
        plotCorrectPlacesFromDF(classicalDFClearer, numPlayers, True, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}classical/clearer/correctPlaces{numPlayers}wErr.png", yRange=None, xlabel="Tournament", ylabel="Proportion")



    renamedSortingDF = filterDF(classicalAndSortingDF, "tournament", sortingNames)
    oldsortingNames = renamedSortingDF["tournament"].values
    newSortingNames = [name[:2] if name[:3] != "BIS" else "BIS" for name in oldsortingNames]
    renamedSortingDF["tournament"] = newSortingNames

    makeAccuracyPlots(renamedSortingDF, "bestOf", f"{rootFolder}sortingBestOf/", True, "Best-of Number", legLocs=["upper left", "upper right", "upper right", "upper left"])

    # plotCorrectPlacesFromCSV("classicalAndSorting")

    # Now the performance tests from and (optimised) MAB tournaments
    RRvsMABDFCollection = []
    for MABSubFolderName in ["", "lowExp/"]:
        if MABSubFolderName == "":
            MABMainDF = pd.read_csv("csvs/MABMainTestsAveraged.csv")
        else:
            MABMainDF = pd.read_csv("csvs/MABMainTestsLowExpAveraged.csv")

        for MABFolderName in ["MABFull", "MAB"]:
            makeAccuracyPlots(MABMainDF, "numPlayers", f"{rootFolder}{MABFolderName}/{MABSubFolderName}", True, "Number of Players", legLocs=["upper right", "upper right", "upper center", "upper right"])

            for numPlayers in MABMainDF["numPlayers"].unique():
                plotCorrectPlacesFromDF(MABMainDF, numPlayers, False, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}{MABFolderName}/{MABSubFolderName}correctPlaces{numPlayers}.png", yRange=None, xlabel="Tournament", ylabel="Proportion")
                plotCorrectPlacesFromDF(MABMainDF, numPlayers, True, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}{MABFolderName}/{MABSubFolderName}correctPlaces{numPlayers}wErr.png", yRange=None, xlabel="Tournament", ylabel="Proportion")

            # And do the same to make graphs that compare RR vs MAB
            # RRDF = filterDF(classicalDF, "tournament", ["RR1", "RR5", "RR25"])#, "RR100"])
            RRDF = filterDF(classicalDF, "tournament", ["RR1", "RR5", "RR25"])#, "RR100"])
            if MABFolderName == "MAB":
                RRDF = filterDF(RRDF, "tournament", ["RR1", "RR5"])

            MABDF = MABMainDF
            for col in MABDF.columns:
                if col not in RRDF.columns:
                    MABDF.drop(columns=[col], inplace=True)
            for col in RRDF.columns:
                if col not in MABDF.columns:
                    RRDF.drop(columns=[col], inplace=True)
            RRvsMABDF = pd.concat([RRDF, MABDF])

            # ####################### HMM ########################
            # RRvsMABDF = filterDF(RRvsMABDF, "numPlayers", [16])
            # ####################### HMM ########################

            makeAccuracyPlots(RRvsMABDF, "numPlayers", f"{rootFolder}RRvs{MABFolderName}/{MABSubFolderName}", True, "Number of Players", legLocs=["upper left", "upper right", "upper center", "upper left"])

            for numPlayers in RRvsMABDF["numPlayers"].unique():
                plotCorrectPlacesFromDF(RRvsMABDF, numPlayers, False, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}RRvs{MABFolderName}/{MABSubFolderName}correctPlaces{numPlayers}.png", yRange=None, xlabel="Tournament", ylabel="Proportion")
                plotCorrectPlacesFromDF(RRvsMABDF, numPlayers, True, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}RRvs{MABFolderName}/{MABSubFolderName}correctPlaces{numPlayers}wErr.png", yRange=None, xlabel="Tournament", ylabel="Proportion")

            # Remove TS n=32 tests for the next loop iteration because they're too innefficient to be interesting
            toDrop = []
            MABMainDF.reset_index(inplace=True)
            for i in range(len(MABMainDF)):
                row = MABMainDF.loc[i]
                if row["tournament"] == "TS" and row["numPlayers"] == 32:
                    toDrop.append(i)
            MABMainDF.drop(toDrop, axis=0, inplace=True)

        RRvsMABDFCollection.append([MABSubFolderName, RRvsMABDF])

    # Now do the same for the transitivity test
    for subFolderName in ["", "lowExp/"]:
        if subFolderName == "":
            transitivityDF = pd.read_csv("csvs/transitivityTestsAveraged.csv")
        else:
            transitivityDF = pd.read_csv("csvs/transitivityTestsLowExpAveraged.csv")
        makeAccuracyPlots(transitivityDF, "strongTransitivity", f"{rootFolder}transitivity/{subFolderName}", True, "Strong Transitivity?", legLocs=["lower left", "upper right", "upper right", "upper left"])
        for numPlayers in transitivityDF["numPlayers"].unique():
            plotCorrectPlacesFromDF(transitivityDF, numPlayers, False, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}transitivity/{subFolderName}correctPlaces{numPlayers}.png", yRange=None, xlabel="Tournament", ylabel="Proportion")
            plotCorrectPlacesFromDF(transitivityDF, numPlayers, True, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}transitivity/{subFolderName}correctPlaces{numPlayers}wErr.png", yRange=None, xlabel="Tournament", ylabel="Proportion")

    # And the fixed number of matches test
    for subFolderName in ["", "lowExp/"]:
        if subFolderName == "":
            fixedMatchNum120DF = pd.read_csv(f"csvs/fixedMatchesTests120Averaged.csv")
            fixedMatchNum2400DF = pd.read_csv(f"csvs/fixedMatchesTests2400Averaged.csv")
            fixedMatchNum12000DF = pd.read_csv(f"csvs/fixedMatchesTests12000Averaged.csv")
        else:
            fixedMatchNum120DF = pd.read_csv(f"csvs/fixedMatchesTestsLowExp120Averaged.csv")
            fixedMatchNum2400DF = pd.read_csv(f"csvs/fixedMatchesTestsLowExp2400Averaged.csv")
            fixedMatchNum12000DF = pd.read_csv(f"csvs/fixedMatchesTestsLowExp12000Averaged.csv")

        fixedMatchNum120DF["fixedMatchNum"] = 120
        fixedMatchNum2400DF["fixedMatchNum"] = 2400
        fixedMatchNum12000DF["fixedMatchNum"] = 12000
        fixedMatchNumDF = pd.concat([fixedMatchNum120DF, fixedMatchNum2400DF, fixedMatchNum12000DF])

        fixedMatchNumDF = filterDF(fixedMatchNumDF, "tournament", ["RR1", "RR5", "RR100", "SE", "DE", "SW", "UCB", "TS", "EG0.2"])
        
        makeAccuracyPlots(fixedMatchNumDF, "fixedMatchNum", f"{rootFolder}fixedMatchNum/{subFolderName}", False, "Number of Matches", legLocs=["upper right", "upper left", "upper center", "lower right"])

        for num, DF in [[120, fixedMatchNum120DF], [2400, fixedMatchNum2400DF], [12000, fixedMatchNum12000DF]]:
            for numPlayers in DF["numPlayers"].unique():
                plotCorrectPlacesFromDF(DF, numPlayers, False, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}fixedMatchNum/{subFolderName}correctPlaces{numPlayers}_{num}.png", yRange=None, xlabel="Tournament", ylabel="Proportion")
                plotCorrectPlacesFromDF(DF, numPlayers, True, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}fixedMatchNum/{subFolderName}correctPlaces{numPlayers}wErr_{num}.png", yRange=None, xlabel="Tournament", ylabel="Proportion")

    # MAB Tuning Tests
    for param, neatParam in [["explorationFolds", "Number of Exploration Folds"], ["patience", "Patience"], ["maxLockInProportion", "Maximum Lock-In Proportion"]]:
        MABTuningDF = pd.read_csv(f"csvs/MABTuningTestsAveraged2_{param}.csv")

        for numPlayers in [8, 32]:
            MABTuningDFFiltered = filterDF(MABTuningDF, "numPlayers", [numPlayers])

            makeAccuracyPlots(MABTuningDFFiltered, param, f"{rootFolder}MABTuning/{numPlayers}/{param}/", True, neatParam, legLocs=["upper right", "upper right", "upper right", "upper right"])

    # MABTuningDF = pd.read_csv("csvs/MABTuningTestsAveraged2.csv")

    # EGNames = []
    # for name in MABTuningDF["tournament"].unique():
    #     if name[:2] == "EG":
    #         EGNames.append(name)

    # EGTuningDF = filterDF(MABTuningDF, "tournament", EGNames)
    # print(EGTuningDF, EGTuningDF["tournament"].unique(), EGTuningDF["numPlayers"].unique())
    # epsilons = []
    # for name in EGTuningDF["tournament"].values:
    #     epsilons.append(name[2:])

    # EGTuningDF["epsilon"] = epsilons
    # EGTuningDF["tournament"] = "EG"

    # EGTuningDF.to_csv("csvs/MABTuningTestsAveraged2_epsilon.csv", index=False)
    # print("written")


    EGTuningDF = pd.read_csv("csvs/MABTuningTestsAveraged2_epsilonAveraged.csv")

    makeAccuracyPlots(EGTuningDF, "numPlayers", f"{rootFolder}MABTuning/epsilon/", True, "Number of Players", "epsilon", legLocs=["upper right", "upper right", "upper right", "upper right"], xlabel="Epsilon")


    # And finally plot graphs comparing ALL tournaments (though not all RRs)...
    classicalNonRRDF = pd.read_csv("csvs/classicalAndSortingTestsAveraged.csv")
    classicalNonRRDF = filterDF(classicalNonRRDF, "tournament", ["SE", "SE3pp", "DE", "SW"])

    for MABSubFolderName, RRvsMAB in RRvsMABDFCollection:
        allDF = pd.concat([RRvsMAB, classicalNonRRDF, sortingDF])

        makeAccuracyPlots(allDF, "numPlayers", f"{rootFolder}ALL/{MABSubFolderName}", True, "Number of Players", legLocs=["upper right", "upper left", "upper left", "upper right"])

        for numPlayers in allDF["numPlayers"].unique():
            plotCorrectPlacesFromDF(allDF, numPlayers, False, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}ALL/{MABSubFolderName}correctPlaces{numPlayers}.png", yRange=None, xlabel="Tournament", ylabel="Proportion")
            plotCorrectPlacesFromDF(allDF, numPlayers, True, title="Proportion Of Tests With Correctly Ranked ith Players", filename=f"{rootFolder}ALL/{MABSubFolderName}correctPlaces{numPlayers}wErr.png", yRange=None, xlabel="Tournament", ylabel="Proportion")

        # ...including comparing Elo rankings to predicted rankings
        allEloDF = allDF.copy()

        ####################### HMM ########################
        allEloDF = filterDF(allEloDF, "numPlayers", [16])
        allDF = filterDF(allDF, "numPlayers", [16])
        ####################### HMM ########################

        allEloDF["rankType"] = "Elo"
        allDF["rankType"] = "Predicted (Default)"

        allEloDF["cosine0"] = allEloDF["eloCosine0"]
        allEloDF["cosine0STD"] = allEloDF["eloCosine0STD"]
        allEloDF["cosine1"] = allEloDF["eloCosine1"]
        allEloDF["cosine1STD"] = allEloDF["eloCosine1STD"]

        allEloVSPred = pd.concat([allDF, allEloDF])
        makeAccuracyPlots(allEloVSPred, "rankType", f"{rootFolder}ALL_Elo/{MABSubFolderName}", True, "Ranking Method", legLocs=["upper right", "upper left", "upper left", "upper right"])


    gridLines = False