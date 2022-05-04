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

headers = ["tournament", "numPlayers", "strongTransitivity", "numMatches", "numRounds", "bestOf",
           "cosine0", "cosine1", "eloCosine0", "eloCosine1", "correctPositions", "eloCorrectPositions"]

outputCSV = "csvs/classicalAndSortingTests.csv"

with open(outputCSV, "a") as f:
    for col in headers:
        f.write(col)
        f.write(",")
    f.write("\n")

def runTournamentForStats(tournament : Tournament, strongTransitivity = False) -> Dict:
    tournament.runAllMatches()

    trueRanking = getTrueRanking(tournament.strengths)

    ranking = tournament.getRanking()
    eloRank, _ = tournament.getEloRanking()

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

def writeStatsCollectionToCSV(statsCollection : List[Dict], filename : str, headers=False):
    with open(filename, "a") as f:
        if headers:
            for col in statsCollection[0]:
                f.write(col)
                f.write(",")
            f.write("\n")

        for stats in statsCollection:
            for key in stats:
                f.write(f"{stats[key]},")
        
            f.write("\n")

bestOfs = [1,3,5,7,9,51,101]
numPlayers = [4,8,16,32,64]
numTests = 1000

for i in range(numTests):
    start = time.time()
    statsCollection = []
    for n in numPlayers:
        strengths = generateStrengths(n)
        tournaments = []

        # Round Robin Tests
        for numFolds in [1,2,3,5,10,25,50,100]:
            tournaments.append(RoundRobin(strengths, numFolds, verbose=False))

        # Single Elimination
        tournaments.append(SingleElimination(strengths, verbose=False))
        tournaments.append(SingleElimination(strengths, thirdPlacePlayoff=True, verbose=False))

        # Double Elimination
        tournaments.append(DoubleElimination(strengths, verbose=False))

        # Swiss
        tournaments.append(Swiss(strengths, verbose=False))

        # Sorting Algos
        for bestOf in bestOfs:
            for algo in [InsertionSort, BinaryInsertionSort, BubbleSort, SelectionSort, QuickSort, MergeSort, HeapSort]:
                tournaments.append(algo(strengths, bestOf=bestOf, verbose=False))

        
        for t in tournaments:
            statsCollection.append(runTournamentForStats(t))

    writeStatsCollectionToCSV(statsCollection, outputCSV)

    print(f"Run {i+1}/{numTests} done in {time.time()-start:.4f}s.")
        

