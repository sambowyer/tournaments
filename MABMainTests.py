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

outputCSV = "csvs/MABMainTestsLowExp.csv"

writeHeaders(outputCSV, headers)

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


# optimalUCBParams = {"explorationFolds" : 3,
#                     "patience": 4,
#                     "maxLockInProportion": 0.25}

# optimalTSParams = {"explorationFolds" : 3,
#                     "patience": 2,
#                     "maxLockInProportion": 0.25}

# optimalEGParams = {"explorationFolds" : 3,
#                     "patience": 4,
#                     "maxLockInProportion": 0.05,
#                     "epsilon": 0.1}

optimalUCBParams = {"explorationFolds" : 1,
                    "patience": 4,
                    "maxLockInProportion": 0.5}

optimalTSParams = {"explorationFolds" : 1,
                    "patience": 2,
                    "maxLockInProportion": 0.5}

optimalEGParams = {"explorationFolds" : 1,
                    "patience": 5,
                    "maxLockInProportion": 0.5,
                    "epsilon": 0.2}

numPlayers = [4,8,16,32,64] 
numTests = 1000

for i in range(numTests):
    start = time.time()
    statsCollection = []
    for n in numPlayers:
        strengths = generateStrengths(n)
        tournaments = []

        # UCB
        tournaments.append(UCB(strengths, explorationFolds=optimalUCBParams["explorationFolds"], patience=optimalUCBParams["patience"], maxLockInProportion=optimalUCBParams["maxLockInProportion"], verbose=False))

        # TS
        if n < 32 or (n == 32 and i % 4 == 0):
            tournaments.append(TS(strengths, explorationFolds=optimalTSParams["explorationFolds"], patience=optimalTSParams["patience"], maxLockInProportion=optimalTSParams["maxLockInProportion"], verbose=False))

        # EG
        tournaments.append(EG(strengths, explorationFolds=optimalEGParams["explorationFolds"], patience=optimalEGParams["patience"], maxLockInProportion=optimalEGParams["maxLockInProportion"], epsilon=optimalEGParams["epsilon"], verbose=False))

        for t in tournaments:
            statsCollection.append(runTournamentForStats(t))

    writeStatsCollectionToCSV(statsCollection, outputCSV)

    print(f"Run {i+1}/{numTests} done in {time.time()-start:.4f}s.")