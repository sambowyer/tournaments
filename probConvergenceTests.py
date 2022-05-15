import time
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from ClassicTournaments import RoundRobin
from MABTournaments import UCB, TS, EG
from utils import *
from typing import List, Dict


numPlayers = 8

folder = f"img/report_images/convergence/{numPlayers}/"
# folder = f"img/report_images/convergence/RR/{numPlayers}/"

def makeFitnessHistoryPlot(xvalues : List[int], yvalues : List[float], lineLabels : List[str], title : str, filename : str, ylabel="Mean Strength Error", xlim=None, legendLoc="upper right", sizeInches=[17,10], extraticks=[]):
    fig, ax = plt.subplots()
    # maxX = max([len(x) for x in xvalues])
    maxX = len(xvalues)
    for i, tournamentValues in enumerate(yvalues):
        # Make sure that each line to be plotted has the same length (pad with np.nan which will be invisible in the plot)
        # print(i)
        ax.plot(xvalues, tournamentValues + [np.nan for j in range(maxX-len(tournamentValues))], label=lineLabels[i])

    ax.set_title(title, fontsize=22)
    ax.set_xlabel("Match Number", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', labelsize=16)

    plt.legend(loc=legendLoc, fontsize=18, title_fontsize=18)

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(min([min(ys[xlim[0]:xlim[1]]) if ys[xlim[0]:xlim[1]]!=[] else 1 for ys in yvalues]), max([max(ys[xlim[0]:xlim[1]]) if ys[xlim[0]:xlim[1]]!=[] else 0 for ys in yvalues]))

    fig.set_size_inches(sizeInches[0], sizeInches[1])
    plt.xticks(list(plt.xticks()[0]) + extraticks)

    plt.savefig(filename, bbox_inches='tight')

    plt.close()

def meanError(matrix1 : np.ndarray, matrix2 : np.ndarray, excludeDiagonal=True) -> float:
    matrix = abs(matrix1-matrix2)
    total  = 0
    count  = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j:
                total += matrix[i,j]
                count += 1
    return total/count
    
numTests = 1000

strengthsCollection = [generateStrengths(numPlayers) for _ in range(numTests)]

def runTournaments(tournamentConstructors, WRL=True) -> List:
    errors = {}
    cosines = {}
    eloCosines = {}
    numMatches = {}

    for i in range(numTests):
        print("Starting")
        start = time.time()
        strengths = strengthsCollection[i]

        trueRanking = getTrueRanking(strengths)

        tournaments = [tc(strengths) for tc in tournamentConstructors]

        for t in tournaments:
            if i == 0:
                errors[t.toString()] = []
                numMatches[t.toString()] = []
                cosines[t.toString()] = []
                eloCosines[t.toString()] = []

            meanErrs = []
            cos = []
            eloCos = []
            while not t.isFinished:
                t.runNextMatch()
                if WRL:
                    err = meanError(strengths, t.winRatesLaplace)
                else:
                    err = meanError(strengths, t.winRates)

                # norm = np.linalg.norm(abs(strengths - t.winRatesLaplace))  # Don't use this as 'err' is probably more intuitive
                # maxE = np.max(abs(strengths - t.winRatesLaplace))  # Don't use this as it'll go down (or up) in steps and not be as smooth as the other two
                # print(strengths, t.winRatesLaplace)
                # print(err, norm, maxE)

                meanErrs.append(err)
                # meanErrs.append(norm)

                ranking = t.getRanking()
                eloRank, _ = t.getEloRanking()

                rankingSimNumSamples = min(120, getNumberOfPossibleDefiniteRankings(ranking))
                eloRankSimNumSamples = min(120, getNumberOfPossibleDefiniteRankings(eloRank))

                # rankingSimNumSamples = 5
                # eloRankSimNumSamples = 5

                cos.append(getRankingSimilarity(trueRanking, ranking, rankingSimNumSamples))
                eloCos.append(getRankingSimilarity(trueRanking, eloRank, eloRankSimNumSamples))
            print(f"Finished {t.toString()}")


            errors[t.toString()].append(meanErrs)
            cosines[t.toString()].append(cos)
            eloCosines[t.toString()].append(eloCos)
            numMatches[t.toString()].append(len(t.schedule))

        print(f"Run {i+1}/{numTests} done in {time.time()-start:.4f}s.")
    
    return errors, cosines, eloCosines, numMatches

if "RR" not in folder:
    # U = lambda strengths:UCB(strengths, verbose=False, explorationFolds=3, patience=4, maxLockInProportion=0.25)
    # T = lambda strengths:TS(strengths, verbose=False, explorationFolds=3, patience=2, maxLockInProportion=0.25)
    # E = lambda strengths:EG(strengths, verbose=False, explorationFolds=3, patience=4, maxLockInProportion=0.05, epsilon=0.1)
    U = lambda strengths:UCB(strengths, verbose=False, explorationFolds=1, patience=4, maxLockInProportion=0.5)
    T = lambda strengths:TS(strengths, verbose=False, explorationFolds=1, patience=2, maxLockInProportion=0.5)
    E = lambda strengths:EG(strengths, verbose=False, explorationFolds=1, patience=5, maxLockInProportion=0.5, epsilon=0.2)
    
    R = lambda strengths: RoundRobin(strengths, 3, verbose=False)

    errors, cosines, eloCosines, numMatches = runTournaments([U,T,E,R])

    maximinX = max([min(numMatches[t]) for t in numMatches])+1
    middleX  = max([min(numMatches[t]) for t in ["UCB", "EG0.2", "RR3"]])

    R2 = lambda strengths: RoundRobin(strengths, math.ceil(maximinX/(0.5*numPlayers*(numPlayers-1))), verbose=False)
    errors2, cosines2, eloCosines2, numMatches2 = runTournaments([R2])

    errors.update(errors2)
    cosines.update(cosines2)
    eloCosines.update(eloCosines2)
    numMatches.update(numMatches2)

    legend = ['UCB', "TS", "EG", "RR3", f"RR{math.ceil(maximinX/(0.5*numPlayers*(numPlayers-1)))}"]
else:
    errors, cosines, eloCosines, numMatches = runTournaments([lambda strengths: RoundRobin(strengths, 200, verbose=False)], False)
    maximinX = max([min(numMatches[t]) for t in numMatches])+1
    middleX  = 500
    legend = ["RR200"]


averagedErrs = []
averagedCosines = []
averagedEloCosines = []

for t in errors:
    averagedErr = []
    averagedCosine = []
    averagedEloCosine = []

    for i in range(min(numMatches[t])):
        numActiveTournaments = 0
        currentTotal = 0
        currentCosTotal = 0
        currentEloCosTotal = 0

        for e in errors[t]:
            if i < len(e):
                numActiveTournaments += 1
                currentTotal += e[i]

        for e in cosines[t]:
            if i < len(e):
                currentCosTotal += e[i]
        
        for e in eloCosines[t]:
            if i < len(e):
                currentEloCosTotal += e[i]

        averagedErr.append(currentTotal/numActiveTournaments)
        averagedCosine.append(currentCosTotal/numActiveTournaments)
        averagedEloCosine.append(currentEloCosTotal/numActiveTournaments)

    averagedErrs.append(averagedErr)
    averagedCosines.append(averagedCosine)
    averagedEloCosines.append(averagedEloCosine)

xVals = np.arange(max([max(numMatches[t]) for t in numMatches])+1)


# newValues = []
# for i, tournamentValues in enumerate(averagedErrs):
#     # Make sure that each line to be plotted has the same length (pad with np.nan which will be invisible in the plot)
#     # print(i)
#     newValues.append(tournamentValues + [np.nan for j in range(maxX-len(tournamentValues))])

# df = pd.DataFrame({"match": xVals, "RR": newValues[0], "UCB": newValues[1],  "TS": newValues[2],  "EG": newValues[3]})
# df.to_csv(f"csvs/probConvergenceTests{numTests}.csv", header=True)


makeFitnessHistoryPlot(xVals, averagedErrs, legend, "Mean Strength Estimation Error Over Time", f"{folder}meanErrorWhole.png", xlim=[0, maximinX+25])
makeFitnessHistoryPlot(xVals, averagedErrs, legend, "Mean Strength Estimation Error Over Time", f"{folder}meanErrorStart.png", xlim=[0, middleX])
makeFitnessHistoryPlot(xVals, averagedErrs, legend, "Mean Strength Estimation Error Over Time", f"{folder}meanErrorEnd.png", xlim=[middleX-100, maximinX+25])#, extraticks=[500])


makeFitnessHistoryPlot(xVals, averagedCosines, legend, "Cosine Similarity Of The Predicted Ranking Over Time", f"{folder}meanCosineWhole.png", ylabel="Cosine Similarity", xlim=[0, maximinX+25], legendLoc="lower right")
makeFitnessHistoryPlot(xVals, averagedCosines, legend, "Cosine Similarity Of The Predicted Ranking Over Time", f"{folder}meanCosineStart.png", ylabel="Cosine Similarity", xlim=[0, middleX], legendLoc="lower right")
makeFitnessHistoryPlot(xVals, averagedCosines, legend, "Cosine Similarity Of The Predicted Ranking Over Time", f"{folder}meanCosineEnd.png", ylabel="Cosine Similarity", xlim=[middleX-100, maximinX+25], legendLoc="lower right")#, extraticks=[500])


makeFitnessHistoryPlot(xVals, averagedEloCosines, legend, "Cosine Similarity Of The Elo Ranking Over Time", f"{folder}meanEloCosineWhole.png", ylabel="Cosine Similarity", xlim=[0, maximinX+25])
makeFitnessHistoryPlot(xVals, averagedEloCosines, legend, "Cosine Similarity Of The Elo Ranking Over Time", f"{folder}meanEloCosineStart.png", ylabel="Cosine Similarity", xlim=[0, middleX])
makeFitnessHistoryPlot(xVals, averagedEloCosines, legend, "Cosine Similarity Of The Elo Ranking Over Time", f"{folder}meanEloCosineEnd.png", ylabel="Cosine Similarity", xlim=[middleX-100, maximinX+25])

