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

headers = ["tournament", "numPlayers", "strongTransitivity", "numMatches", "numRounds", "bestOf", "explorationFolds", "patience", "maxLockInProportion",
           "cosine0", "cosine1", "eloCosine0", "eloCosine1", "correctPositions", "eloCorrectPositions"]

outputCSV = "csvs/MABTuningTests.csv"

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
             "explorationFolds": tournament.explorationFolds, 
             "patience": tournament.patience,
             "maxLockInProportion": tournament.maxLockInProportion,
             "cosine0": getRankingSimilarity(trueRanking, ranking, rankingSimNumSamples),#[0], 
             "cosine1": getRankingSimilarity(trueRanking, ranking, rankingSimNumSamples, False),#[0],
             "eloCosine0": getRankingSimilarity(trueRanking, eloRank, eloRankSimNumSamples),#[0],
             "eloCosine1": getRankingSimilarity(trueRanking, eloRank, eloRankSimNumSamples, False),#[0], 
             "correctPositions": str(list(proportionCorrectPositionsVector(trueRanking, ranking))).strip("[]").replace(", ", "_"),
             "eloCorrectPositions": str(list(proportionCorrectPositionsVector(trueRanking, eloRank))).strip("[]").replace(", ", "_")}

    return stats

explorationFolds = [0,1,2,3]
patience = [2,3,4,5]
maxLockInProportion = [0.05,0.1,0.25,0.5]
epsilons = [0.01,0.05,0.1,0.2]
numPlayers = [8,32]
numTests = 49

for i in range(numTests):
    start = time.time()
    statsCollection = []
    for n in numPlayers:
        strengths = generateStrengths(n)
        tournaments = []

        for exp in explorationFolds:
            for pat in patience:
                for mlp in maxLockInProportion:
                    # UCB
                    tournaments.append(UCB(strengths, explorationFolds=exp, patience=pat, maxLockInProportion=mlp, verbose=False))

                    # TS - can easily take FOREVER to finish
                    if n == 8 or (pat < 4 and mlp < 0.25 and exp in (1,3)):
                        tournaments.append(TS(strengths, explorationFolds=exp, patience=pat, maxLockInProportion=mlp, verbose=False))

                    # EG
                    for eps in epsilons:
                        tournaments.append(EG(strengths, explorationFolds=exp, patience=pat, maxLockInProportion=mlp, epsilon=eps, verbose=False))

        for j, t in enumerate(tournaments):
            statsCollection.append(runTournamentForStats(t))
            # print(j, len(t.schedule))

    writeStatsCollectionToCSV(statsCollection, outputCSV)

    print(f"Run {i+1}/{numTests} done in {time.time()-start:.4f}s.")