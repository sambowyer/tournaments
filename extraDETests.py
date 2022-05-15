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

headers = ["tournament", "numPlayers", "strongTransitivity", "numMatches", "numRounds", "bestOf",
           "cosine0", "cosine1", "eloCosine0", "eloCosine1", "correctPositions", "eloCorrectPositions"]

outputCSV = "csvs/classicalAndSortingTestsFIXED.csv"

# writeHeaders(outputCSV, headers)

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
             "cosine0": getRankingSimilarity(trueRanking, ranking, rankingSimNumSamples),#[0], 
             "cosine1": getRankingSimilarity(trueRanking, ranking, rankingSimNumSamples, False),#[0],
             "eloCosine0": getRankingSimilarity(trueRanking, eloRank, eloRankSimNumSamples),#[0],
             "eloCosine1": getRankingSimilarity(trueRanking, eloRank, eloRankSimNumSamples, False),#[0], 
             "correctPositions": str(list(proportionCorrectPositionsVector(trueRanking, ranking))).strip("[]").replace(", ", "_"),
             "eloCorrectPositions": str(list(proportionCorrectPositionsVector(trueRanking, eloRank))).strip("[]").replace(", ", "_")}
    print(stats["cosine1"])
    return stats

numPlayers = [64]
numTests = 447

for i in range(numTests):
    start = time.time()
    statsCollection = []
    for n in numPlayers:
        found = False
        while not found:
            strengths = generateStrengths(n)
            stats = runTournamentForStats(DoubleElimination(strengths, verbose=False))
            if stats["cosine1"] != 0:
                found = True
        print(stats["cosine1"])
        statsCollection.append(stats)

    writeStatsCollectionToCSV(statsCollection, outputCSV)

    print(f"Run {i+1}/{numTests} done in {time.time()-start:.4f}s.")
        