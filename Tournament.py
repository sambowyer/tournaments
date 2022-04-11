import numpy as np
from typing import List
import random

class Tournament:
    def __init__(self, strengths : np.ndarray, eloFunc=lambda x: 1/(1+10**(x/400)), bestOf=1):
        self.strengths = strengths 
        self.eloFunc   = eloFunc
        self.bestOf    = bestOf

        self.isFinished = False

        self.numPlayers  = strengths.shape[0]
        self.schedule    = []  # records all the matches that have been played up to this point
        self.resultsList = []  # records all the results (each 0 or 1) of all matches that have already been played

        self.resultsMatrix    = np.zeros([self.numPlayers, self.numPlayers])
        self.eloScores        = [1000 for i in range(self.numPlayers)]
        self.eloScoresHistory = [self.eloScores.copy()]
        self.winRates         = np.zeros([self.numPlayers, self.numPlayers])
        self.winRatesLaplace  = np.array([[0 if x==y else 0.5 for x in range(self.numPlayers)] for y in range(self.numPlayers)])

        self.verbose = True

    def getNextMatch(self) -> List[int]:
        '''Should return a size-2 list containing the two competitors (i.e. 2 ints) playing the next match. If no more matches in schedule, then return None.
        This is perhaps the most important way to distinguish between different tournament systems.'''
        return [0,1]

    def runNextMatch(self):
        '''Runs the next match, updates the schedule, resultsList, resultsMatrix, eloScores, winRates, winRatesLaplace.'''
        match = self.getNextMatch()
        
        if match is None:
            self.isFinished = True
        else:
            result = self.getMatchResult(match)
            self.updateStats(match, result)

    def getMatchResult(self, match : List[int]):
        cumulativeScore = 0
        for _ in range(self.bestOf):
            outcome = random.random()

            # We set result to 0 if match[0] wins, and 1 if match[1] wins
            singleResult = 0 if outcome <= self.strengths[match[0], match[1]] else 1
            cumulativeScore += singleResult

        result = 0 if cumulativeScore < self.bestOf/2 else 1

        return result

    def updateStats(self, match : List[int], result : int):
        self.schedule.append(match)
        self.resultsList.append(result)

        # These might help to follow along with the rest of this method
        winner = match[result]
        loser  = match[1-result]

        if self.verbose:
            print(f"{winner} beat {loser}")
        
        # Now to update the state of the tournament
        # First update the results matrix
        self.resultsMatrix[winner, loser] += 1

        # Then the Elo scores
        eloDifference      = self.eloScores[winner] - self.eloScores[loser]
        pointsTransfer     = self.eloFunc(eloDifference)
        self.eloScores[winner] += 25*pointsTransfer
        self.eloScores[loser]  -= 25*pointsTransfer

        self.eloScoresHistory.append(self.eloScores.copy())

        # Finally the win rates matrix and the win rates matrix with Laplace succession
        self.winRates[winner, loser] = self.resultsMatrix[winner, loser] / (self.resultsMatrix[winner, loser] + self.resultsMatrix[loser, winner])
        self.winRates[loser, winner] = self.resultsMatrix[loser, winner] / (self.resultsMatrix[winner, loser] + self.resultsMatrix[loser, winner])

        self.winRatesLaplace[winner, loser] = (1 + self.resultsMatrix[winner, loser]) / (2 + self.resultsMatrix[winner, loser] + self.resultsMatrix[loser, winner])
        self.winRatesLaplace[loser, winner] = (1 + self.resultsMatrix[loser, winner]) / (2 + self.resultsMatrix[winner, loser] + self.resultsMatrix[loser, winner])     

    def runAllMatches(self):
        '''This method will be overridden in many of the tournament systems where it doesn't make sense to have a getNextMatch() method (e.g. in sorting algorithm-based tournaments)'''
        while not self.isFinished:
            self.runNextMatch()
    
    def getTotalWins(self) -> List[int]:
        '''Returns a list of the total number of wins that each player has achieved.'''
        return [sum(x) for x in self.resultsMatrix]

    def getTotalWinsHistory(self) -> List[List[int]]:
        '''Returns a list of lists of the total wins of each player after each match is played.'''
        currentTotalWins = [0 for i in range(self.numPlayers)]
        history = [currentTotalWins.copy()]
        for i, m in enumerate(self.schedule):
            currentTotalWins[m[self.resultsList[i]]] += 1
            history.append(currentTotalWins.copy())
        return history
    
    def getRanking(self) -> List[int]:
        '''Returns a ranked list of the players according to the specifics of the tournament system.
        MUST HAVE self.isFinished==True.'''
        pass 

    def getTotalWinRanking(self) -> (List[int], List[int]):
        '''Returns a ranked list of the players according to the total number of wins as well as a list with the corresponding number of wins for each player.'''
        ranking = range(self.numPlayers)
        totalWins = self.getTotalWins()
        ranking = sorted(ranking, key = lambda x: totalWins[x], reverse=True)

        return ranking, sorted(totalWins, reverse=True)

    def getEloRanking(self) -> (List[int], List[float]):
        '''Returns a ranked list of the players according to their Elo scores as well as a list with the corresponding Elo scores for each player.'''
        ranking = range(self.numPlayers)
        ranking = sorted(ranking, key = lambda x: self.eloScores[x], reverse=True)

        return ranking, sorted(self.eloScores, reverse=True)

    def getAverageWinRateRanking(self) -> (List[int], List[float]):
        '''Returns a ranked list of the players according to their average win-rates as well as a list with the corresponding win-rates for each player.'''
        ranking = range(self.numPlayers)
        # numGames = [0 for i in range(self.numPlayers)]
        # for i in range(self.numPlayers):
        #     for j in range(self.numPlayers):
        #         numGames[i] = self.resultsMatrix[i,j] + self.resultsMatrix[j,i]
        numGames = [sum(self.resultsMatrix[x]) + sum(self.resultsMatrix[:,x]) for x in range(self.numPlayers)]
        averageWinRate = [sum(self.resultsMatrix[x])/numGames[x] for x in range(self.numPlayers)]

        ranking = sorted(ranking, key = lambda x: averageWinRate[x], reverse=True)

        return ranking, sorted(averageWinRate, reverse=True)

    def getAverageWinRateLaplaceRanking(self) -> (List[int], List[float]):
        '''Returns a ranked list of the players according to their average win-rates (with Laplace's rule of succession) as well as a list with the corresponding win-rates-with-succession for each player.'''
        ranking = range(self.numPlayers)
        # numGames = [0 for i in range(self.numPlayers)]
        # for i in range(self.numPlayers):
        #     for j in range(self.numPlayers):
        #         numGames[i] = self.resultsMatrix[i,j] + self.resultsMatrix[j,i]
        numGames = [sum(self.resultsMatrix[x]) + sum(self.resultsMatrix[:,x]) for x in range(self.numPlayers)]
        averageWinRate = [(1+sum(self.resultsMatrix[x]))/(2+numGames[x]) for x in range(self.numPlayers)]

        ranking = sorted(ranking, key = lambda x: averageWinRate[x], reverse=True)

        return ranking, sorted(averageWinRate, reverse=True)
