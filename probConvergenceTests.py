import time
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from ClassicTournaments import RoundRobin
from MABTournaments import UCB, TS, EG
from utils import *
from typing import List, Dict

def makeFitnessHistoryPlot(xvalues : List[int], yvalues : List[float], lineLabels : List[str], title : str, filename : str):
    fig, ax = plt.subplots()
    # maxX = max([len(x) for x in xvalues])
    maxX = len(xvalues)
    for i, tournamentValues in enumerate(yvalues):
        # Make sure that each line to be plotted has the same length (pad with np.nan which will be invisible in the plot)
        # print(i)
        ax.plot(xvalues, tournamentValues + [np.nan for j in range(maxX-len(tournamentValues))], label=lineLabels[i])

    ax.set_title(title)
    ax.set_xlabel("Match Number")
    ax.set_ylabel("Mean strength error")
    plt.legend()

    plt.show()

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
    
numTests = 50

errors = {}
numMatches = {}

for i in range(numTests):
    start = time.time()
    strengths = generateStrengths(16)
    R = RoundRobin(strengths, 200, verbose=False)
    U = UCB(strengths, verbose=False, explorationFolds=3, patience=4, maxLockInProportion=0.25)
    T = TS(strengths, verbose=False, explorationFolds=3, patience=4, maxLockInProportion=0.25)
    E = EG(strengths, verbose=False, explorationFolds=3, patience=4, maxLockInProportion=0.25, epsilon=0.1)

    for t in [R, U, T, E]:
        if i == 0:
            errors[t.toString()] = []
            numMatches[t.toString()] = []

        meanErrs = []
        while not t.isFinished:
            t.runNextMatch()
            err = meanError(strengths, t.winRatesLaplace)

            norm = np.linalg.norm(abs(strengths - t.winRatesLaplace))  # Don't use this as 'err' is probably more intuitive
            # maxE = np.max(abs(strengths - t.winRatesLaplace))  # Don't use this as it'll go down (or up) in steps and not be as smooth as the other two
            # print(strengths, t.winRatesLaplace)
            # print(err, norm, maxE)

            meanErrs.append(err)
            # meanErrs.append(norm)

        errors[t.toString()].append(meanErrs)
        numMatches[t.toString()].append(len(t.schedule))
    print(f"Run {i+1}/{numTests} done in {time.time()-start:.4f}s.")

averagedErrs = []
for t in errors:
    averaged = []
    for i in range(min(numMatches[t])):
        numActiveTournaments = 0
        currentTotal = 0
        for e in errors[t]:
            if i < len(e):
                numActiveTournaments += 1
                currentTotal += e[i]
        averaged.append(currentTotal/numActiveTournaments)
    averagedErrs.append(averaged)

xVals = np.arange(max([max(numMatches[t]) for t in numMatches])+1)
maxX = len(xVals)
newValues = []
for i, tournamentValues in enumerate(averagedErrs):
    # Make sure that each line to be plotted has the same length (pad with np.nan which will be invisible in the plot)
    # print(i)
    newValues.append(tournamentValues + [np.nan for j in range(maxX-len(tournamentValues))])

df = pd.DataFrame({"match": xVals, "RR": newValues[0], "UCB": newValues[1],  "TS": newValues[2],  "EG": newValues[3]})
df.to_csv(f"csvs/probConvergenceTests{numTests}.csv", header=True)

makeFitnessHistoryPlot(xVals, averagedErrs, list(errors.keys()), "ooh", "ahh.png")