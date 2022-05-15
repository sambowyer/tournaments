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

outputCSV = "csvs/fixedMatchesTestsLowExp2400.csv"

writeHeaders(outputCSV, headers)

def runTournamentForStats(strengths, tournamentConstructor, constructorArgs : List, maxNumMatches : int, strongTransitivity = False) -> Dict:
    trueRanking = getTrueRanking(strengths)

    rankings = []
    eloRanks = []
    numMatches = 0
    numRounds = 0
    while numMatches < maxNumMatches:
        tournament = tournamentConstructor(*constructorArgs)
        # print(tournament.toString())
        tournament.runAllMatches()
        rankings.append(tournament.getRanking())
        eloRanks.append(tournament.getEloRanking()[0])
        # print(len(tournament.schedule))
        numMatches += len(tournament.schedule)
        numRounds  += tournament.getNumRounds()

    if tournament.toString()[:2] in ["UC", "TS", "EG", "DE"]:
        if len(rankings) != 1:
            # print(numMatches - len(tournament.schedule))
            numMatches -= len(tournament.schedule)
            rankings = rankings[1:]
            eloRanks = eloRanks[1:]

    # print(tournament.toString(), len(rankings), numMatches)

    ranking = combineRankings(rankings)
    eloRank = combineRankings(eloRanks)

    rankingSimNumSamples = min(120, getNumberOfPossibleDefiniteRankings(ranking))
    eloRankSimNumSamples = min(120, getNumberOfPossibleDefiniteRankings(eloRank))

    stats = {"tournament": tournament.toString(),
             "numPlayers": tournament.numPlayers,
             "strongTransitivity": strongTransitivity,
             "numMatches": numMatches,
             "numRounds": numRounds,
             "bestOf": tournament.bestOf,
             "cosine0": getRankingSimilarity(trueRanking, ranking, rankingSimNumSamples),#[0], 
             "cosine1": getRankingSimilarity(trueRanking, ranking, rankingSimNumSamples, False),#[0],
             "eloCosine0": getRankingSimilarity(trueRanking, eloRank, eloRankSimNumSamples),#[0],
             "eloCosine1": getRankingSimilarity(trueRanking, eloRank, eloRankSimNumSamples, False),#[0], 
             "correctPositions": str(list(proportionCorrectPositionsVector(trueRanking, ranking))).strip("[]").replace(", ", "_"),
             "eloCorrectPositions": str(list(proportionCorrectPositionsVector(trueRanking, eloRank))).strip("[]").replace(", ", "_")}

    return stats

numPlayers = [16]
numTests = 500

numTotalMatches = 2400

for i in range(numTests):
    start = time.time()
    statsCollection = []
    
    tournaments = []

    for n in numPlayers:
        strengths = generateStrengths(n)

        # Round Robin Tests
        for numFolds in [1,5]:#,25,100]:
            statsCollection.append(runTournamentForStats(strengths, RoundRobin, [strengths, numFolds, lambda x: 1/(1+10**(x/400)), 1, False], numTotalMatches))

        # Single Elimination
        statsCollection.append(runTournamentForStats(strengths, SingleElimination, [strengths, False, lambda x: 1/(1+10**(x/400)), 1, False], numTotalMatches))

        # tournaments.append(SingleElimination(strengths, thirdPlacePlayoff=True, verbose=False))

        # Double Elimination
        statsCollection.append(runTournamentForStats(strengths, DoubleElimination, [strengths, lambda x: 1/(1+10**(x/400)), 1, False], numTotalMatches))

        # Swiss
        statsCollection.append(runTournamentForStats(strengths, Swiss, [strengths, lambda x: 1/(1+10**(x/400)), 1, False], numTotalMatches))

        # UCB
        # statsCollection.append(runTournamentForStats(strengths, UCB, [strengths, lambda x: 1/(1+10**(x/400)), 1, False, 3, 4, 0.25, numTotalMatches], numTotalMatches))
        statsCollection.append(runTournamentForStats(strengths, UCB, [strengths, lambda x: 1/(1+10**(x/400)), 1, False, 1, 4, 0.5, numTotalMatches], numTotalMatches))

        # TS
        # statsCollection.append(runTournamentForStats(strengths, TS, [strengths, lambda x: 1/(1+10**(x/400)), 1, False, 3, 2, 0.25, numTotalMatches], numTotalMatches))
        statsCollection.append(runTournamentForStats(strengths, TS, [strengths, lambda x: 1/(1+10**(x/400)), 1, False, 1, 2, 0.5, numTotalMatches], numTotalMatches))

        # EG
        # statsCollection.append(runTournamentForStats(strengths, EG, [strengths, lambda x: 1/(1+10**(x/400)), 1, False, 3, 4, 0.05, 0.1, numTotalMatches], numTotalMatches))
        statsCollection.append(runTournamentForStats(strengths, EG, [strengths, lambda x: 1/(1+10**(x/400)), 1, False, 1, 5, 0.5, 0.2, numTotalMatches], numTotalMatches))


    writeStatsCollectionToCSV(statsCollection, outputCSV)

    print(f"Run {i+1}/{numTests} done in {time.time()-start:.4f}s.")