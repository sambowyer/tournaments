import numpy as np
from typing import List
import random
from utils import *
from testsUtil import *
from Tournament import Tournament 
from ClassicTournaments import *
from SortingTournaments import *
from MABTournaments import *
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import time

def runTournamentXTimes(tournament : Tournament, X : int):
    '''Runs a tournament X number of times and returns the combined ranks from those given by the tournaments and those given by Elo scores'''
    rankings = []
    eloRankings = []

    for _ in range(X):
        tournament.runAllMatches()

        rankings.append(tournament.getRanking())
        eloRankings.append(tournament.getEloRanking()[0])

    return combineRankings(rankings), combineRankings(eloRankings)

headers = ["tournament", "numPlayers", "strongTransitivity", "numMatches", "numRounds", "bestOf",
           "cosine0", "cosine1", "eloCosine0", "eloCosine1", "correctPositions", "eloCorrectPositions"]

outputCSV = "csvs/transitivityTests.csv"

writeHeaders(outputCSV, headers)

def runTournamentForStats(tournament : Tournament, strongTransitivity = False, numTimesToRun=1) -> Dict:
    trueRanking = getTrueRanking(tournament.strengths)

    ranking, eloRank = runTournamentXTimes(tournament, numTimesToRun)

    rankingSimNumSamples = min(120, getNumberOfPossibleDefiniteRankings(ranking))
    eloRankSimNumSamples = min(120, getNumberOfPossibleDefiniteRankings(eloRank))

    stats = {"tournament": tournament.toString(),
             "numPlayers": tournament.numPlayers,
             "strongTransitivity": strongTransitivity,
             "numMatches": len(tournament.schedule),
             "numRounds": tournament.getNumRounds(),
             "bestOf": tournament.bestOf,
             "cosine0": getRankingSimilarity(trueRanking, ranking, rankingSimNumSamples)[0], 
             "cosine1": getRankingSimilarity(trueRanking, ranking, rankingSimNumSamples, False)[0],
             "eloCosine0": getRankingSimilarity(trueRanking, eloRank, eloRankSimNumSamples)[0],
             "eloCosine1": getRankingSimilarity(trueRanking, eloRank, eloRankSimNumSamples, False)[0], 
             "correctPositions": str(list(proportionCorrectPositionsVector(trueRanking, ranking))).strip("[]").replace(", ", "_"),
             "eloCorrectPositions": str(list(proportionCorrectPositionsVector(trueRanking, eloRank))).strip("[]").replace(", ", "_")}

    return stats

optimalUCBParams = {"exlplorationFolds" : None,
                    "patience": None,
                    "maxLockInProportion": None}

optimalTSParams = {"exlplorationFolds" : None,
                    "patience": None,
                    "maxLockInProportion": None}

optimalEGParams = {"exlplorationFolds" : None,
                    "patience": None,
                    "maxLockInProportion": None,
                    "epsilon": None}

numPlayers = [4,8,16,32,64]
numTests = 1000

numTimesToRun = []  # needs to contain the number of times each tournament should run (in the order that they are created in the loop below)
# MAB need to be treated differently and need to just run for some fixed number of matches --- not too complicated really though --- their entries in this list should be 1

for i in range(numTests):
    start = time.time()
    statsCollection = []
    
    tournaments = []

    for n in numPlayers:
        strengths = generateStrengths(n)

        # Round Robin Tests
        for numFolds in [1,5,25,100]:
            tournaments.append(RoundRobin(strengths, numFolds, verbose=False))

        # Single Elimination
        tournaments.append(SingleElimination(strengths, verbose=False))
        tournaments.append(SingleElimination(strengths, thirdPlacePlayoff=True, verbose=False))

        # Double Elimination
        tournaments.append(DoubleElimination(strengths, verbose=False))

        # Swiss
        tournaments.append(Swiss(strengths, verbose=False))

        # Sorting Algos
        for algo in [InsertionSort, BinaryInsertionSort, BubbleSort, SelectionSort, QuickSort, MergeSort, HeapSort]:
            tournaments.append(algo(strengths, verbose=False))

        # UCB
        tournaments.append(UCB(strengths, explorationFolds=optimalUCBParams["explorationFolds"], patience=optimalUCBParams["patience"], maxLockInProportion=optimalUCBParams["maxLockInProportion"], verbose=False))

        # TS - too large a patience will result in TS taking FOREVER to finish
        tournaments.append(TS(strengths, explorationFolds=optimalTSParams["explorationFolds"], patience=optimalTSParams["patience"], maxLockInProportion=optimalTSParams["maxLockInProportion"], verbose=False))

        # EG
        tournaments.append(EG(strengths, explorationFolds=optimalEGParams["explorationFolds"], patience=optimalEGParams["patience"], maxLockInProportion=optimalEGParams["maxLockInProportion"], epsilon=optimalEGParams["epsilon"], verbose=False))


        for j, t in enumerate(tournaments):
            statsCollection.append(runTournamentForStats(t, numTimesToRun=numTimesToRun[j]))

    writeStatsCollectionToCSV(statsCollection, outputCSV)

    print(f"Run {i+1}/{numTests} done in {time.time()-start:.4f}s.")