import numpy as np
from typing import List
import random
from utils import *
from Tournament import Tournament 
from ClassicTournaments import *
from SortingTournaments import *
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

def runTournament(tournament : Tournament, graphStep=1, title="", filename="", id=0, verbose=True):
    # totalWinsHistory = []
    # eloScoresHistory = []

    # while not tournament.isFinished:
    #     totalWinsHistory.append(tournament.getTotalWins())
    #     eloScoresHistory.append(tournament.eloScores.copy())

    #     tournament.runNextMatch()
    tournament.runAllMatches()

    # TODO: Clean up this monstrosity (USE PANDAS DATAFRAMES)
    totalWinsHistory = tournament.getTotalWinsHistory()
    eloScoresHistory = tournament.eloScoresHistory

    strPlayers, strNumbers   = getAverageStrengthRanking(tournament.strengths)
    winPlayers, winNumbers   = tournament.getTotalWinRanking()
    eloPlayers, eloNumbers   = tournament.getEloRanking()
    awrPlayers, awrNumbers   = tournament.getAverageWinRateRanking()
    awrlPlayers, awrlNumbers = tournament.getAverageWinRateLaplaceRanking()
    wrPlayers, wrNumbers     = getDominationDegreeRanking(tournament.winRates)
    wrlPlayers, wrlNumbers     = getDominationDegreeRanking(tournament.winRatesLaplace)

    for arr in (strNumbers, eloNumbers, awrNumbers, awrlNumbers):
        for i in range(len(arr)):
            if arr[i] is not None:
                arr[i] = f"{arr[i]:.3f}"

    trueRanking = getTrueRanking(tournament.strengths)
    # domRanking, _ = getDominationDegreeRanking(tournament.strengths)
    # print(trueRanking)
    # print(domRanking)
    predictedRanking = tournament.getRanking()

    # print("Rank |  True   Predicted   Player (Str)   Player (Wins)   Player (Elos)   Player (AWR)   Player (AWRL)")
    # print("------------------------------------------------------------------------------------------------------")
    # for i in range(tournament.numPlayers):
    #     print(f" {i+1:2}  |   {trueRanking[i]:2}       {predictedRanking[i]:2}       {strPlayers[i]:2} ({strNumbers[i]:.2f})      {winPlayers[i]:2} ({winNumbers[i]:4})       {eloPlayers[i]:2} ({eloNumbers[i]:7.2f})    {awrPlayers[i]:2} ({awrNumbers[i]:.2f})      {awrlPlayers[i]:2} ({awrlNumbers[i]:.2f})")
        
    # print(f"Cosine similarities to true ranking:\nPred: {getRankingSimilarity(trueRanking, predictedRanking)}\nStrs: {getRankingSimilarity(trueRanking, strPlayers)}\nWins: {getRankingSimilarity(trueRanking, winPlayers)}\nElos: {getRankingSimilarity(trueRanking, eloPlayers)}")

    df = pd.DataFrame({"Pos.": ["1st", "2nd", "3rd"]+[f"{i}th" for i in range(4, len(trueRanking)+1)],
                       "True" : trueRanking,
                       f"Pred ({getRankingSimilarity(trueRanking, predictedRanking)[0]:.4f})" : predictedRanking, 
                       f"Wins ({getRankingSimilarity(trueRanking, winPlayers)[0]:.4f})" : zip(winPlayers, winNumbers), 
                       f"Elo ({getRankingSimilarity(trueRanking, eloPlayers)[0]:.4f})" : zip(eloPlayers, eloNumbers), 
                       f"Avg Str ({getRankingSimilarity(trueRanking, strPlayers)[0]:.4f})" : zip(strPlayers, strNumbers), 
                       f"Avg WR ({getRankingSimilarity(trueRanking, awrPlayers)[0]:.4f})" : zip(awrPlayers, awrNumbers), 
                       f"Avg WRL ({getRankingSimilarity(trueRanking, awrlPlayers)[0]:.4f})" : zip(awrlPlayers, awrlNumbers), 
                       f"WR Dom ({getRankingSimilarity(trueRanking, wrPlayers)[0]:.4f})" : zip(wrPlayers, wrNumbers), 
                       f"WRL Dom ({getRankingSimilarity(trueRanking, wrlPlayers)[0]:.4f})" : zip(wrlPlayers, wrlNumbers)})
    df.set_index("Pos.", inplace=True)
    # print(df.to_string())
    # df.drop(labels=["Avg WR", "Avg WRL"], axis=1, inplace=True)
    print(title)
    print(tabulate(df, headers="keys", tablefmt="github"))

    totalWinsHistory = totalWinsHistory[::graphStep]
    eloScoresHistory = eloScoresHistory[::graphStep]
    gameNumLabels    = np.arange(len(totalWinsHistory))*graphStep

    fig = plt.figure(num=id, clear=True)

    ax1 =plt.subplot(1,2,1)
    for player in range(tournament.numPlayers):
        plt.plot(gameNumLabels, [x[player] for x in totalWinsHistory])

    ax1.set(xlabel='Game #', ylabel='Number of wins')
    plt.legend(range(tournament.numPlayers), loc="upper left")

    ax2 =plt.subplot(1,2,2)
    for player in range(tournament.numPlayers):
        plt.plot(gameNumLabels, [x[player] for x in eloScoresHistory])

    ax2.set(xlabel='Game #', ylabel='Elo rating')
    
    fig.suptitle(title)

    fig.set_size_inches(15, 9)
    
    if filename == "":
        plt.show()
    else:
        plt.savefig(filename)

    f = plt.figure()
    f.clear()
    plt.close(f)

def runTournamentTest():
    strengths = generateStrengths(8)

    count = 0

    RR = RoundRobin(strengths, 10)
    runTournament(RR, title="Round Robin (10-fold)", filename="img/wins and elo plots/RR.png", id=count)
    count += 1

    SE = SingleElimination(strengths)
    runTournament(SE, title="Single Elimination", filename="img/wins and elo plots/SE.png", id=count)
    count += 1

    DE = DoubleElimination(strengths)
    runTournament(DE, title="Double Elimination", filename="img/wins and elo plots/DE.png", id=count)
    count += 1

    SW = Swiss(strengths)
    runTournament(SW, title="Swiss Style", filename="img/wins and elo plots/SW.png", id=count)
    count += 1

    IS = InsertionSort(strengths, bestOf=7)
    runTournament(IS, title="Insertion Sort", filename="img/wins and elo plots/IS.png", id=count)
    count += 1

    BIS = BinaryInsertionSort(strengths, bestOf=7)
    runTournament(BIS, title="Binary Insertion Sort", filename="img/wins and elo plots/BIS.png", id=count)
    count += 1

    BS = BubbleSort(strengths, bestOf=7)
    runTournament(BS, title="Bubble Sort", filename="img/wins and elo plots/BS.png", id=count)
    count += 1

    SS = SelectionSort(strengths, bestOf=7)
    runTournament(SS, title="Selection Sort", filename="img/wins and elo plots/SS.png", id=count)
    count += 1

    QS = QuickSort(strengths, bestOf=7)
    runTournament(QS, title="Quick Sort", filename="img/wins and elo plots/QS.png", id=count)
    count += 1

    MS = MergeSort(strengths, bestOf=7)
    runTournament(MS, title="Merge Sort", filename="img/wins and elo plots/MS.png", id=count)
    count += 1

    HS = HeapSort(strengths, bestOf=7)
    runTournament(HS, title="Heap Sort", filename="img/wins and elo plots/HS.png", id=count)
    count += 1

def makeBarChart(xticks : List[str], yvalues : List, title : str, xlabel : str, ylabel : str, filename : str, yRange=None, yError=None):
    fig, ax = plt.subplots()
    rects1 = ax.bar(xticks, yvalues)
    if yError is not None:
        ax.errorbar(xticks, yvalues, yerr = yError,fmt='o',ecolor = 'red',color='yellow')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yRange is not None:
        ax.set_ylim(yRange)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)

    # fig.tight_layout()
    fig.set_size_inches(17, 10)

    plt.savefig(filename)
    plt.close()

def makeDoubledBarChart(xticks : List[str], yvalues1 : List, yvalues2 : List, ylabel1 : str, ylabel2 : str, title : str, xlabel : str, ylabel : str, filename : str, yRange=None, yErrors=None):
    fig, ax = plt.subplots()
    x = np.arange(len(xticks))
    width=0.35
    rects1 = ax.bar(x-width/2, yvalues1, width, label=ylabel1)
    rects2 = ax.bar(x+width/2, yvalues2, width, label=ylabel2)
    if yErrors is not None:
        ax.errorbar(x-width/2, yvalues1, yerr = yErrors[0],fmt='o',ecolor = 'red',color='yellow')
        ax.errorbar(x+width/2, yvalues2, yerr = yErrors[1],fmt='o',ecolor = 'red',color='yellow')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.legend()
    if yRange is not None:
        ax.set_ylim(yRange)

    
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)


    autolabel(rects1)
    autolabel(rects2)

    # fig.tight_layout()

    fig.set_size_inches(17, 10)

    plt.savefig(filename)
    plt.close()

def prelimTest():
    n = 16
    numGames = 100

    tournamentNames = ["RR1", "RR10", "RR100", "SE", "DE", "SW", "IS", "BIS", "BS", "SS", "QS", "MS", "HS", "IS7", "BIS7", "BS7", "SS7", "QS7", "MS7", "HS7"]
    stats = {}
    for t in tournamentNames:
        stats[t] = {"correctPlaces" : np.zeros(n), "cosine" : 0, "cosineSq" : 0, "eloCorrectPlaces" : np.zeros(n), "eloCosine" : 0, "eloCosineSq" : 0,  "numMatches" : 0}

    for runNum in range(numGames):
        strengths = generateStrengths(n)
        trueRanking = getTrueRanking(strengths)

        RR1   = RoundRobin(strengths, 1)
        RR10  = RoundRobin(strengths, 10)
        RR100 = RoundRobin(strengths, 100)
        SE    = SingleElimination(strengths)
        DE    = DoubleElimination(strengths)
        SW    = Swiss(strengths)
        IS    = InsertionSort(strengths)
        BIS   = BinaryInsertionSort(strengths)
        BS    = BubbleSort(strengths)
        SS    = SelectionSort(strengths)
        QS    = QuickSort(strengths)
        MS    = MergeSort(strengths)
        HS    = HeapSort(strengths)
        IS7   = InsertionSort(strengths, bestOf=7)
        BIS7  = BinaryInsertionSort(strengths, bestOf=7)
        BS7   = BubbleSort(strengths, bestOf=7)
        SS7   = SelectionSort(strengths, bestOf=7)
        QS7   = QuickSort(strengths, bestOf=7)
        MS7   = MergeSort(strengths, bestOf=7)
        HS7   = HeapSort(strengths, bestOf=7)

        tournaments = [RR1, RR10, RR100, SE, DE, SW, IS, BIS, BS, SS, QS, MS, HS, IS7, BIS7, BS7, SS7, QS7, MS7, HS7]

        for i, tournament in enumerate(tournaments):
            tournament.verbose = False
            tournament.runAllMatches()

            predictedRanking = tournament.getRanking()
            eloRanking, _    = tournament.getEloRanking()
            
            stats[tournamentNames[i]]["numMatches"]  += len(tournament.schedule)
            cosineSim = getRankingSimilarity(trueRanking, predictedRanking, numSamples=30)[0]
            eloCosineSim = getRankingSimilarity(trueRanking, eloRanking, numSamples=30)[0]
            stats[tournamentNames[i]]["cosine"]      += cosineSim
            stats[tournamentNames[i]]["eloCosine"]   += eloCosineSim
            stats[tournamentNames[i]]["cosineSq"]    += cosineSim**2
            stats[tournamentNames[i]]["eloCosineSq"] += eloCosineSim**2
            stats[tournamentNames[i]]["correctPlaces"]    += proportionCorrectPositionsVector(trueRanking, predictedRanking)
            stats[tournamentNames[i]]["eloCorrectPlaces"] += proportionCorrectPositionsVector(trueRanking, eloRanking)
            # if i==2:
            #     print(stats[tournamentNames[i]]["correctPlaces"], stats[tournamentNames[i]]["eloCorrectPlaces"])

            # for j, place in enumerate(trueRanking):
            #     if predictedRanking[j] == place:
            #         stats[tournamentNames[i]]["correctPlaces"][j] += 1
            #     if eloRanking[j] == place:
            #         stats[tournamentNames[i]]["eloCorrectPlaces"][j] += 1
        
        print(f"Run {runNum+1}/{numGames } done.")

    print()

    for i, tournament in enumerate(tournaments):
        stats[tournamentNames[i]]["numMatches"] /= numGames
        stats[tournamentNames[i]]["cosine"]     /= numGames
        stats[tournamentNames[i]]["eloCosine"]  /= numGames
        stats[tournamentNames[i]]["cosineSq"]     /= numGames
        stats[tournamentNames[i]]["eloCosineSq"]  /= numGames
        stats[tournamentNames[i]]["correctPlaces"]    /= numGames
        stats[tournamentNames[i]]["eloCorrectPlaces"] /= numGames
        
        # for j in range(n):
        #     stats[tournamentNames[i]]["correctPlaces"][j]    /= numGames
        #     stats[tournamentNames[i]]["eloCorrectPlaces"][j] /= numGames

        # print(f"{tournamentNames[i]}\n {stats[tournamentNames[i]]}\n")

    
    validTournamentNames = ["RR1", "RR10", "RR100", "SE", "DE", "SW", "IS", "IS7", "BIS", "BIS7", "BS", "BS7", "SS", "SS7", "QS", "QS7", "MS", "MS7", "HS", "HS7"]

    cosineStds    = [math.sqrt(stats[t]["cosineSq"]-stats[t]["cosine"]**2) for t in validTournamentNames]
    eloCosineStds = [math.sqrt(stats[t]["eloCosineSq"]-stats[t]["eloCosine"]**2) for t in validTournamentNames]

    cosineStdsScaled    = [math.sqrt(stats[t]["cosineSq"]/stats[t]["numMatches"]-(stats[t]["cosine"]/stats[t]["numMatches"])**2) for t in validTournamentNames]
    eloCosineStdsScaled = [math.sqrt(stats[t]["eloCosineSq"]/stats[t]["numMatches"]-(stats[t]["eloCosine"]/stats[t]["numMatches"])**2) for t in validTournamentNames]

    # print(cosineStds, eloCosineStds)

    # Make bar charts for all the correct-place-number stats
    for i in range(n):
        predRankStats = [stats[t]["correctPlaces"][i] for t in validTournamentNames]
        eloRankStats  = [stats[t]["eloCorrectPlaces"][i] for t in validTournamentNames]
        makeDoubledBarChart(validTournamentNames, predRankStats, eloRankStats, "Predicted Ranking", "Elo Ranking", f"Proportion of players correctly ranked #{i+1}", "Tournament", "Proportion", f"img/correctRank{i+1}.png", yRange=[0,1])

    # Make cosine distance graphs
    predRankCosines = [stats[t]["cosine"] for t in validTournamentNames]
    eloRankCosines  = [stats[t]["eloCosine"] for t in validTournamentNames]
    makeDoubledBarChart(validTournamentNames, predRankCosines, eloRankCosines, "Predicted Ranking", "Elo Ranking", f"Average cosine similarity to true ranking", "Tournament", "Similarity", "img/cosine.png", yRange=[0.99*min([min(predRankCosines), min(eloRankCosines)]),1.01])
    makeDoubledBarChart(validTournamentNames, predRankCosines, eloRankCosines, "Predicted Ranking", "Elo Ranking", f"Average cosine similarity to true ranking", "Tournament", "Similarity", "img/cosineWithStdBars.png", yRange=[0.99*min([min(predRankCosines), min(eloRankCosines)]),1.01], yErrors=[cosineStds, eloCosineStds])
    makeDoubledBarChart(validTournamentNames, cosineStds, eloCosineStds, "Predicted Ranking", "Elo Ranking", f"Standard deviation of similarity to true ranking", "Tournament", "Similarity", "img/cosineStd.png")


    # Make numMatches graphs
    numMatches = [stats[t]["numMatches"] for t in validTournamentNames]
    makeBarChart(validTournamentNames, [stats[t]["numMatches"] for t in validTournamentNames], "Average number of matches played per tournament", "Tournament", "# Matches", "img/numMatches.png")

    # print(numMatches)

    # Make scaled cosine distance graphs
    predRankCosines = [predRankCosines[i]/numMatches[i] for i in range(len(validTournamentNames))]
    eloRankCosines  = [eloRankCosines[i]/numMatches[i] for i in range(len(validTournamentNames))]
    makeDoubledBarChart(validTournamentNames, predRankCosines, eloRankCosines, "Predicted Ranking", "Elo Ranking", f"Average cosine similarity to true ranking divided by mean number of matches in tournament", "Tournament", "Similarity / # Matches", "img/cosineScaled.png")
    makeDoubledBarChart(validTournamentNames, predRankCosines, eloRankCosines, "Predicted Ranking", "Elo Ranking", f"Average cosine similarity to true ranking divided by mean number of matches in tournament", "Tournament", "Similarity / # Matches", "img/cosineScaledWithStdBars.png", yErrors=[cosineStdsScaled, eloCosineStdsScaled])
    makeDoubledBarChart(validTournamentNames, cosineStdsScaled, eloCosineStdsScaled, "Predicted Ranking", "Elo Ranking", f"Standard deviation of average cosine similarity to true ranking divided by mean number of matches in tournament", "Tournament", "Similarity / # Matches", "img/cosineScaledStd.png")

# runTournamentTest()
# prelimTest()

for _ in range(10):
    strengths = generateStrengths(8)
    # print(strengths)
    # SE = SingleElimination(strengths, verbose=True)
    # runTournament(SE)
    # SE = SingleElimination(strengths, thirdPlacePlayoff=True, verbose=True)
    # runTournament(SE)
    DE = DoubleElimination(strengths, verbose=True)
    runTournament(DE)
    # SE = SingleElimination(strengths, thirdPlacePlayoff=True, verbose=True)
    # runTournament(SE)

#     RR = RoundRobin(strengths, 50, verbose=False)
#     runTournament(RR)