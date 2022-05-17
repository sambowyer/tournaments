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

outputCSV = "csvs/RRDomTest.csv"

writeHeaders(outputCSV, headers)

def runTournamentForStats(tournament : Tournament, strongTransitivity = False) -> Dict:
    tournament.runAllMatches()

    trueRanking = getTrueRanking(tournament.strengths)

    ranking = tournament.getRanking()
    eloRank, _ = tournament.getEloRanking()

    rankingSimNumSamples = min(120, getNumberOfPossibleDefiniteRankings(ranking))
    eloRankSimNumSamples = min(120, getNumberOfPossibleDefiniteRankings(eloRank))

    # rankingSimNumSamples = 120
    # eloRankSimNumSamples = 120

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

    return stats

numPlayers = 16
numTests = 1000

for i in range(numTests):
    start = time.time()
    statsCollection = []

    strengths = generateStrengths(numPlayers)
    # print("strengths made")

    tournaments = []

    # Round Robin Tests
    for numFolds in [1,5,10,25,50,75, 100]:
        tournaments.append(RoundRobin(strengths, numFolds, verbose=False))
        tournaments.append(RRDom(strengths, numFolds, verbose=False))

    for t in tournaments:
        statsCollection.append(runTournamentForStats(t))
        # print(t.toString(), len(t.schedule))

    writeStatsCollectionToCSV(statsCollection, outputCSV)

    print(f"Run {i+1}/{numTests} done in {time.time()-start:.4f}s.")
        

