import numpy as np 
import random
import math
import matplotlib.pyplot as plt
from collections import deque
from typing import List

def generateStrengths(numPlayers : int) -> np.ndarray:
    '''Returns a square numpy array of relative strengths between players.'''

    strengths = np.zeros([numPlayers,numPlayers])
    tempStrength = 0
    for i in range(numPlayers):
        for j in range(i+1, numPlayers):
            transitive = False
            while not transitive:
                transitive = True
                tempStrength = random.uniform(0,1)
                if tempStrength != 0.5:
                    for k in range(numPlayers):
                        if k not in [i,j]:
                            if strengths[i,k] > 0.5 and strengths[k,j] > 0.5 and tempStrength < 0.5:
                                transitive = False
                                break
            strengths[i,j] = tempStrength
            strengths[j,i] = 1-tempStrength
            
    return strengths

def getStrengthsSubmatrix(strengths: np.ndarray, validPlayers : List[int]) -> np.ndarray:
    resultsTemp = [i in validPlayers for i in range(strengths.shape[0])]  # so True if i won, false if i lost
    nextRoundStrengths = np.compress(resultsTemp, strengths, axis=0)
    nextRoundStrengths = np.compress(resultsTemp, nextRoundStrengths, axis=1)
    return nextRoundStrengths

def getTrueRanking(strengths : np.ndarray) -> List[int]:
    '''Returns the 'true' ranking of players in 'strengths' (i.e. the longest path in the dominance graph given by 'strengths')'''

    n = strengths.shape[0]
    inDegrees = [(i, sum([1 if (strengths[i,j] < 0.5 and not i==j) else 0 for j in range(n)])) for i in range(n)]
    inDegrees = sorted(inDegrees, key = lambda x: x[1])
    
    return [x[0] for x in inDegrees]

def getAverageStrengthRanking(strengths) -> (List[int], List[float]):
    ranking = range(strengths.shape[0])
    averageStrengths = [sum(x)/(strengths.shape[0]-1) for x in strengths]
    ranking = sorted(ranking, key = lambda x: averageStrengths[x], reverse = True)

    return ranking, sorted(averageStrengths, reverse=True)

def cosineDistance(vec1 : List[int], vec2 : List[int]) -> float:
    '''Returns the cosine distance between two vectors'''
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def getPositionsVector(ranking : List[int]) -> List[int]:
    '''Takes a 'ranking' of players in order from winner to loser and returns a vector whose ith element is the position of the ith player in the ranking.'''
    return [ranking.index(i) for i in range(len(ranking))]

def getRankingSimilarity(ranking1 : List[int], ranking2 : List[int]) -> float:
    return cosineDistance(getPositionsVector(ranking1), getPositionsVector(ranking2))

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

class RoundRobin(Tournament):
    def __init__(self, strengths, numFolds, eloFunc=lambda x: 1/(1+10**(x/400)), bestOf=1):
        super().__init__(strengths, eloFunc, bestOf)
        self.numFolds = numFolds
        
        self.matchesToBePlayed = self.generateAllMatches()

    def getNextMatch(self) -> List[int]:
        if len(self.matchesToBePlayed) == 0:
            return None 
        else:
            return self.matchesToBePlayed.popleft()

    def generateAllMatches(self):
        matches = []
        for _ in range(self.numFolds):
            for x in range(self.numPlayers):
                for y in range(x+1, self.numPlayers):
                    matches.append([x,y])
        random.shuffle(matches)
        return deque(matches)

    def getRanking(self) -> (List[int], List[int]):
        if self.isFinished:
            return self.getTotalWinRanking()[0]
        else:
            print("Not finished yet.")  #TODO: change this

class SingleEliminationRound(Tournament):
    '''This will be able to run a SINGLE round from a single-elimination tournament'''
    def __init__(self, strengths, players, eloFunc=lambda x: 1/(1+10**(x/400))):
        super().__init__(strengths, eloFunc)
        # IMPORTANT: numPlayers must be a power of two
        self.players = players
        self.matches = self.generateMatches(players)
        self.matchNo = 0

    def generateMatches(self, validPlayers : List[int]) -> List[List[int]]:
        '''Pair the players up to generate the matches of this round.'''
        if len(validPlayers) == 1:
            return [None]  # No more matches to be played - we've found the winner
        else:
            random.shuffle(validPlayers)
            nextRoundNumPlayers = int(len(validPlayers)/2)

            return [[validPlayers[i], validPlayers[nextRoundNumPlayers+i]] for i in range(nextRoundNumPlayers)]

    def getNextMatch(self) -> List[int]:
        if self.matchNo >= len(self.matches):  # i.e. if the round has finished
            self.isFinished = True
            return None
        else:
            self.matchNo += 1
            return self.matches[self.matchNo-1]

    def getWinners(self, results : List[int]) -> List[int]:
        winners = []
        for i, m in enumerate(self.matches):
            winners.append(m[results[i]])
        return winners

    def getLosers(self, results : List[int]) -> List[int]:
        losers = []
        for i, m in enumerate(self.matches):
            losers.append(m[1-results[i]])
        return losers

class SingleElimination2(Tournament):
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400)), bestOf=1):
        super().__init__(strengths, eloFunc, bestOf)
        self.currentRound = SingleEliminationRound(strengths, list(range(self.numPlayers)), eloFunc)

    def getNextMatch(self) -> List[int]:
        nextMatch = self.currentRound.getNextMatch()
        if nextMatch is None:  # Round has finished
            winners = self.currentRound.getWinners(self.resultsList[-self.currentRound.matchNo:])
            
            if len(winners) == 1:  # Tournament has finished
                return None
                
            nextRoundStrengths = getStrengthsSubmatrix(strengths, winners)

            self.currentRound = SingleEliminationRound(nextRoundStrengths, winners, self.eloFunc)
        
            return self.getNextMatch()

        return nextMatch

    def getRanking(self) -> (List[int], List[int]):
        if self.isFinished:
            return self.getTotalWinRanking()[0]
        else:
            print("Not finished yet.")  #TODO: change this

class DoubleElimination2(Tournament):
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400)), bestOf=1):
        super().__init__(strengths, eloFunc, bestOf)
        self.currentWinnerRound = SingleEliminationRound(strengths, list(range(self.numPlayers)), eloFunc)
        self.currentLoserRound  = None

        self.newLosers = []

        self.currentTree = "winner"  # either "winner", "loser" (if both the winner and loser trees are finished then we can just supply the actual matches without constructing a Round object)

        self.isWinnerTreeFinished = False
        self.isLoserOnlyRound     = True  # This happens every other round in the loser tree

        self.successfulLosers = None

        self.winnerTreeWinner = None
        self.loserTreeWinner  = None

        self.wc = 0
        self.lc = 0

    def getNextMatch(self) -> List[int]:
        if self.currentTree == "winner":
            nextMatch = self.currentWinnerRound.getNextMatch()
            if nextMatch is None:  # Round has finished
                winners = self.currentWinnerRound.getWinners(self.resultsList[-self.currentWinnerRound.matchNo:])

                self.newLosers = self.currentWinnerRound.getLosers(self.resultsList[-self.currentWinnerRound.matchNo:])

                if len(winners) == 1:  # Tournament has finished
                    self.winnerTreeWinner = winners[0]
                    self.isWinnerTreeFinished = True
                    
                nextRoundStrengths = getStrengthsSubmatrix(strengths, winners)
                self.currentWinnerRound = SingleEliminationRound(nextRoundStrengths, winners, self.eloFunc)
                self.currentTree = "loser"

                with open("aa.txt", "a") as f:
                    self.wc += 1
                    f.write(f"wc {self.wc} {len(self.resultsList)}\n")

                return self.getNextMatch()

            return nextMatch

        elif self.currentTree == "loser":
        
            if self.isLoserOnlyRound:
                if self.currentLoserRound is None:
                    nextRoundStrengths = getStrengthsSubmatrix(strengths, self.newLosers)
                    self.currentLoserRound = SingleEliminationRound(nextRoundStrengths, self.newLosers)

                nextMatch = self.currentLoserRound.getNextMatch()
                if nextMatch is None:  # Round has finished

                    self.successfulLosers = self.currentLoserRound.getWinners(self.resultsList[-self.currentLoserRound.matchNo:])
                    self.currentTree = "winner"
                    self.isLoserOnlyRound = False

                    self.currentLoserRound = None

                    with open("aa.txt", "a") as f:
                        self.lc += 1
                        f.write(f"lc {self.lc} {len(self.resultsList)}\n")

                    return self.getNextMatch()

                return nextMatch

            else:
                if self.currentLoserRound is None:
                    
                    nextRoundStrengths = getStrengthsSubmatrix(strengths, self.newLosers+self.successfulLosers)
                    self.currentLoserRound = SingleEliminationRound(nextRoundStrengths, self.newLosers+self.successfulLosers)

                nextMatch = self.currentLoserRound.getNextMatch()
                if nextMatch is None:  # Round has finished

                    self.successfulLosers = self.currentLoserRound.getWinners(self.resultsList[-self.currentLoserRound.matchNo:])

                    if len(self.successfulLosers) == 1:
                        self.loserTreeWinner = self.successfulLosers[0]
                        self.currentTree = "endgame"
                        return [self.winnerTreeWinner, self.loserTreeWinner]

                    self.isLoserOnlyRound = True

                    # Set up the the next round (which purely consists of the winners of this round - i.e. we stay working in the losers tree)
                    nextRoundStrengths = getStrengthsSubmatrix(strengths, self.successfulLosers)
                    self.currentLoserRound = SingleEliminationRound(nextRoundStrengths, self.successfulLosers)

                    with open("aa.txt", "a") as f:
                        self.lc += 1
                        f.write(f"lc {self.lc} {len(self.resultsList)}\n")

                    return self.getNextMatch()

                return nextMatch
        
        elif self.currentTree == "endgame":
            if self.resultsList[-1] == 0:  # so winnerTreeWinner beat loserTreeWinner (i.e. we're done)
                return None
            else:
                self.currentTree = None
                return [self.winnerTreeWinner, self.loserTreeWinner]
    
        else:
            return None

    def getRanking(self) -> (List[int], List[int]):
        if self.isFinished:
            return self.getTotalWinRanking()[0]
        else:
            print("Not finished yet.")  #TODO: change this

class SingleElimination(Tournament):
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400))):
        super().__init__(strengths, eloFunc)
        # IMPORTANT: numPlayers must be a power of two
        self.matchNoThisRound = 0
        self.currentRoundMatches = self.generateCurrentRoundMatches(list(range(self.numPlayers)))

    def generateCurrentRoundMatches(self, validPlayers : list) -> List[List[int]]:
        '''Given a list of the player's still in the game, pair them up to generate the matches for the next round.'''
        if len(validPlayers) == 1:
            return [None]  # No more matches to be played - we've found the winner
        else:
            random.shuffle(validPlayers)
            nextRoundNumPlayers = int(len(validPlayers)/2)

            return [[validPlayers[i], validPlayers[nextRoundNumPlayers+i]] for i in range(nextRoundNumPlayers)]

    def getPrevRoundWinners(self) -> List[int]:
        winners = []
        # print(f"m={self.matchNoThisRound}")
        for i in range(self.matchNoThisRound):
            # reverse self.resultsList and self.currentRoundMatches so we only consider the matches from the last round
            # print(i)
            winners.append(self.currentRoundMatches[::-1][i][self.resultsList[::-1][i]]) 
        return winners

    def getPrevRoundLosers(self) -> List[int]:
        losers = []
        for i in range(self.matchNoThisRound):
            # reverse self.resultsList and self.currentRoundMatches so we only consider the matches from the last round
            losers.append(self.currentRoundMatches[::-1][i][1-self.resultsList[::-1][i]]) 
        return losers

    def getNextMatch(self) -> List[int]:
        if self.matchNoThisRound >= len(self.currentRoundMatches):  # i.e. if the current round has finished
            # Need to figure out which players won in the last round
            winners = self.getPrevRoundWinners()

            # start the new round with the winners of the previous round
            self.currentRoundMatches = self.generateCurrentRoundMatches(winners)
            self.matchNoThisRound = 0

            return self.getNextMatch()
        else:
            self.matchNoThisRound += 1
            return self.currentRoundMatches[self.matchNoThisRound-1]

    def getRanking(self) -> (List[int], List[int]):
        # Players will only progress to next round if they win the previous round, so we can rank players by who many games they've won
        if self.isFinished:
            return self.getTotalWinRanking() 
        else:
            print("Not finished yet.")  #TODO: change this

class DoubleElimination(Tournament):
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400))):
        super().__init__(strengths, eloFunc)
        # IMPORTANT: numPlayers must be a power of two

        self.mainTree  = SingleElimination(strengths)
        self.loserTree = None

        self.mainTreeLosersHistory = []  # To keep track of the losing players in successive rounds of the main tree

        self.loserTreeRoundNumber = 0

    def getNextMatch(self) -> List[int]:
        '''Works by running through the mainTree tournament in its entirety, keeping track of losers in each round, and then constructing the loser's tree from this information.'''
        self.mainTree.resultsList = self.resultsList.copy()
        if self.mainTree.matchNoThisRound >= len(self.mainTree.currentRoundMatches):  # I.e. if the current round has finished in the main tree
            # Need to figure out which players won in the last round
            print(self.mainTree.resultsList)

            mainTreeWinners = self.mainTree.getPrevRoundWinners()

            self.mainTreeLosersHistory.append(self.mainTree.getPrevRoundLosers())

            # Start the new round with the winners of the previous round
            self.mainTree.currentRoundMatches = self.mainTree.generateCurrentRoundMatches(mainTreeWinners)

            nextMainTreeMatch = self.mainTree.getNextMatch()
            if nextMainTreeMatch is not None:
                self.mainTree.matchNoThisRound = 1
                return nextMainTreeMatch

            else:  # I.e. the main tree is finished
                if self.loserTree is None:
                    # Create the loser tree using just the losers from the main tree in the first round
                    self.loserTree = SingleElimination(strengths)
                    self.loserTree.currentRoundMatches = self.loserTree.generateCurrentRoundMatches(self.mainTreeLosersHistory[0])
                    self.loserTreeRoundNumber += 1
                    

                elif self.loserTree.matchNoThisRound >= len(self.loserTree.currentRoundMatches):  # I.e. if the current round has finished in the main tree
                    # Create a new loser tree with matches generated from the winners of the loser tree and losers of the main tree
                    self.loserTree = SingleElimination(strengths)
                    self.loserTree.currentRoundMatches = self.generateLoserTreeMatches(self.loserTree.getPrevRoundWinners(), self.mainTreeLosersHistory[self.loserTreeRoundNumber].copy())
                    # Increment self.loserTreeRoundNumber
                    self.loserTreeRoundNumber += 1

                self.mainTree.resultsList = self.resultsList.copy()
                return self.loserTree.getNextMatch()
        else:
            self.mainTree.matchNoThisRound += 1
            print(self.mainTree.currentRoundMatches)
            return self.mainTree.currentRoundMatches[self.mainTree.matchNoThisRound-1]

    def generateLoserTreeMatches(self, validPlayers1, validPlayers2) -> List[List[int]]:
        '''Create matches by pairing up players form validPlayers1 (winners from the loser tree) and validPlayers2 (losers from the main tree).'''
        random.shuffle(validPlayers1)
        random.shuffle(validPlayers2)

        print(len(validPlayers1), len(validPlayers2))

        return [[validPlayers1[i], validPlayers2[i]] for i in range(len(validPlayers2))]

    def getRanking(self) -> (List[int], List[int]):
        # TODO: Should main-tree wins be worth more than loser-tree wins in the overall ranking?
        if self.isFinished:
            return self.getTotalWinRanking() 
        else:
            print("Not finished yet.")  #TODO: change this

class Swiss(Tournament):
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400)), bestOf=1):
        super().__init__(strengths, eloFunc, bestOf)

        self.ranking = list(range(self.numPlayers)) 
        random.shuffle(self.ranking)  # probably not necessary to shuffle them but will do it just in case since strength generation of each player isn't entirely independent

    def runAllMatches(self):
        for _ in range(int(math.log2(self.numPlayers))):
            tempRanking = self.ranking.copy()
            roundSchedule = []

            while len(tempRanking) != 0:
                x = tempRanking[0]
                for i, y in enumerate(tempRanking[1:]):
                    if [x,y] not in self.schedule and [y,x] not in self.schedule:
                        roundSchedule.append([x,y])
                        tempRanking = tempRanking[1:i+1] + tempRanking[i+2:]
                        break
            
            # print(roundSchedule)

            for m in roundSchedule:
                result = self.getMatchResult(m)
                self.updateStats(m, result)

            totalWins = self.getTotalWins()
            self.ranking = sorted(self.ranking, key = lambda x: totalWins[x], reverse=True)
            # we do this rather than:
            #   self.ranking, _ = self.getTotalWinRanking()
            # since getTotalWinRanking() places players with the same number of wins in (ascending) order of their names 

            # print(self.ranking)

    def getRanking(self) -> (List[int], List[int]):
        return self.ranking

class SortingAlgorithm(Tournament):
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400)), bestOf=1):
        super().__init__(strengths, eloFunc, bestOf)

        self.ranking = list(range(self.numPlayers))
        random.shuffle(self.ranking)  # probably not necessary to shuffle them but will do it just in case since strength generation of each player isn't entirely independent

    def swapPlayerRanks(self, index1 : int, index2 : int):
        temp = self.ranking[index1]
        self.ranking[index1] = self.ranking[index2]
        self.ranking[index2] = temp

    def getRanking(self) -> List[int]:
        return self.ranking

class InsertionSort(SortingAlgorithm):
    def runAllMatches(self):
        '''Will run through the whole tournament (i.e. algorithm) running each comparison as a match with self.getMatchResult() and self.updateStats()'''
        newRanking = [self.ranking[0]]
        for x in self.ranking[1:]:
            inserted = False
            for i, y in enumerate(newRanking):
                result = self.getMatchResult([x,y])
                self.updateStats([x,y], result)
                if result == 0:
                    newRanking.insert(i, x)
                    inserted = True
                    break
            if not inserted:
                newRanking.append(x)
        self.ranking = newRanking

class BinaryInsertionSort(SortingAlgorithm):
    def runAllMatches(self):
        '''Will run through the whole tournament (i.e. algorithm) running each comparison as a match with self.getMatchResult() and self.updateStats()'''
        # print(self.ranking)
        newRanking = [self.ranking[0]]
        for x in self.ranking[1:]:
            # print(newRanking)
            n    = len(newRanking)
            low  = 0
            high = n
            while low < high:
                mid  = (high+low)//2
                result = self.getMatchResult([x, newRanking[mid]])
                self.updateStats([x, newRanking[mid]], result)
                if result == 0:
                    high = mid
                else:
                    low  = mid+1
                # print(f"x={x}, l={low}, m={mid}, h={high}")

            newRanking.insert(high, x)
        self.ranking = newRanking
        # print(self.ranking)

class BubbleSort(SortingAlgorithm):
    def runAllMatches(self):
        '''Will run through the whole tournament (i.e. algorithm) running each comparison as a match with self.getMatchResult() and self.updateStats()'''
        n = self.numPlayers
        while n > 1:
            m = 0
            for i in range(1, n):
                x = self.ranking[i-1]
                y = self.ranking[i]

                result = self.getMatchResult([x,y])
                self.updateStats([x,y], result)

                if result == 1:
                    # swap the two players
                    self.swapPlayerRanks(i-1, i)
                    # temp = x
                    # self.ranking[i-1] = y
                    # self.ranking[i]   = temp
                    m = i
            n = m

class SelectionSort(SortingAlgorithm):
    def runAllMatches(self):
        '''Will run through the whole tournament (i.e. algorithm) running each comparison as a match with self.getMatchResult() and self.updateStats()'''
        for i in range(self.numPlayers-1):
            # print(self.ranking)
            k = i
            for j in range(i+1, self.numPlayers):
                x = self.ranking[k]
                y = self.ranking[j]

                result = self.getMatchResult([x,y])
                self.updateStats([x,y], result)

                if result == 1:
                    k = j

            # swap the two players at index k and index i
            self.swapPlayerRanks(k, i)
            # temp = self.ranking[k]
            # self.ranking[k] = self.ranking[i]
            # self.ranking[i] = temp
        # print(self.ranking)

class QuickSort(SortingAlgorithm):
    def runAllMatches(self):
        '''Will run through the whole tournament (i.e. algorithm) running each comparison as a match with self.getMatchResult() and self.updateStats()'''
        # print(self.ranking)
        self.ranking = self.quicksort(self.ranking, 0, self.numPlayers-1)

    def quicksort(self, arr : List[int], left : int, right : int) -> List[int]:
        if left < right:
            pivotIndex, arr = self.partition(arr, left, right)
            # print(arr)
            arr = self.quicksort(arr, left, pivotIndex-1)
            arr = self.quicksort(arr, pivotIndex+1, right)
        return arr

    def partition(self, arr : List[int], left : int, right : int) -> (int, List[int]):
        pivotIndex = (left+right+1)//2
        pivot      = arr[pivotIndex]
        
        winners = []
        losers  = []

        for i in range(left, right+1):
            if i == pivotIndex:
                losers = [arr[i]] + losers
            else:
                result = self.getMatchResult([pivot, arr[i]])
                self.updateStats([pivot, arr[i]], result)

                if result == 0:
                    losers.append(arr[i])
                else:
                    winners.append(arr[i])

        return right-len(losers)+1, arr[:left] + winners + losers + arr[right+1:]

class MergeSort(SortingAlgorithm):
    def runAllMatches(self):
        '''Will run through the whole tournament (i.e. algorithm) running each comparison as a match with self.getMatchResult() and self.updateStats()'''
        print(self.ranking)
        self.ranking = self.mergesort(self.ranking)

    def mergesort(self, arr : List[int]) -> List[int]:
        if len(arr) <= 1:
            return arr
        
        left  = self.mergesort(arr[:len(arr)//2])
        right = self.mergesort(arr[len(arr)//2:])

        return self.merge(left, right)

    def merge(self, left : List[int], right : List[int]) -> List[int]:
        arr = []

        while len(left) > 0 and len(right) > 0:
            l = left[0]
            r = right[0]

            result = self.getMatchResult([l, r])
            self.updateStats([l, r], result)

            if result == 0:
                arr.append(l)
                left = left[1:]
            else:
                arr.append(r)
                right = right[1:]

        if len(left) != 0:
            arr += left
        if len(right) != 0:
            arr += right

        print(arr)
        return arr

class HeapSort(SortingAlgorithm):
    def runAllMatches(self):
        '''Will run through the whole tournament (i.e. algorithm) running each comparison as a match with self.getMatchResult() and self.updateStats()'''
        # print(self.ranking)
        self.ranking = self.heapsort(self.ranking, self.numPlayers)[1:]

    def heapsort(self, arr : List[int], n : int) -> List[int]:
        arr = [-1] + arr
        arr = self.toHeap(arr, n)
        for i in range(n, 1, -1):
            temp   = arr[1]
            arr[1] = arr[i]
            arr[i] = temp
            arr = self.bubbleDown(1, arr, i-1)
        return arr

    def toHeap(self, arr : List[int], n : int) -> List[int]:
        for i in range(n//2, 0, -1):
            arr = self.bubbleDown(i, arr, n)
        return arr

    def bubbleDown(self, i : int, heap : List[int], n : int) -> List[int]:
        # print(i, heap, n)
        if self.heapLeft(i) > n:
            return heap
        elif self.heapRight(i) > n:
            x = heap[i]
            y = heap[self.heapLeft(i)]

            result = self.getMatchResult([x,y])
            self.updateStats([x,y], result)

            if result == 0:
                # swap the two players
                temp = heap[i]
                heap[i] = heap[self.heapLeft(i)]
                heap[self.heapLeft(i)] = temp
                # self.swapPlayerRanks(i, self.heapLeft(i))
        else:
            l = heap[self.heapLeft(i)]
            r = heap[self.heapRight(i)]
            x = heap[i]

            resultLR = self.getMatchResult([l,r])
            self.updateStats([l,r], resultLR)

            resultLX = self.getMatchResult([l,x])
            self.updateStats([l,x], resultLX)

            if resultLR == 1 and resultLX == 1:
                temp = heap[i]
                heap[i] = heap[self.heapLeft(i)]
                heap[self.heapLeft(i)] = temp
                # self.swapPlayerRanks(i, self.heapLeft(i))
                heap = self.bubbleDown(self.heapLeft(i), heap, n)
            else:
                resultRX = self.getMatchResult([r,x])
                self.updateStats([r,x], resultRX)

                if resultRX == 1:
                    temp = heap[i]
                    heap[i] = heap[self.heapRight(i)]
                    heap[self.heapRight(i)] = temp
                    # self.swapPlayerRanks(i, self.heapRight(i))
                    heap = self.bubbleDown(self.heapRight(i), heap, n)

        return heap

    def heapLeft(self, i : int) -> int:
        return 2*i

    def heapRight(self, i : int) -> int:
        return 2*i + i

def runTournament(tournament : Tournament, graphStep=1):
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

    trueRanking = getTrueRanking(tournament.strengths)
    predictedRanking = tournament.getRanking()

    print("Rank |  True   Predicted   Player (Str)   Player (Wins)   Player (Elos)   Player (AWR)   Player (AWRL)")
    print("------------------------------------------------------------------------------------------------------")
    for i in range(tournament.numPlayers):
        print(f" {i+1:2}  |   {trueRanking[i]:2}       {predictedRanking[i]:2}       {strPlayers[i]:2} ({strNumbers[i]:.2f})      {winPlayers[i]:2} ({winNumbers[i]:4})       {eloPlayers[i]:2} ({eloNumbers[i]:7.2f})    {awrPlayers[i]:2} ({awrNumbers[i]:.2f})      {awrlPlayers[i]:2} ({awrlNumbers[i]:.2f})")
        
    print(f"Cosine similarities to true ranking:\nPred: {getRankingSimilarity(trueRanking, predictedRanking)}\nStrs: {getRankingSimilarity(trueRanking, strPlayers)}\nWins: {getRankingSimilarity(trueRanking, winPlayers)}\nElos: {getRankingSimilarity(trueRanking, eloPlayers)}")

    totalWinsHistory = totalWinsHistory[::graphStep]
    eloScoresHistory = eloScoresHistory[::graphStep]
    gameNumLabels    = np.arange(len(totalWinsHistory))*graphStep

    ax =plt.subplot(1,2,1)
    for player in range(tournament.numPlayers):
        plt.plot(gameNumLabels, [x[player] for x in totalWinsHistory])

    ax.set(xlabel='Game #', ylabel='Number of wins')
    plt.legend(range(tournament.numPlayers), loc="upper left")

    ax =plt.subplot(1,2,2)
    for player in range(tournament.numPlayers):
        plt.plot(gameNumLabels, [x[player] for x in eloScoresHistory])

    ax.set(xlabel='Game #', ylabel='Elo rating')
    
    plt.show()

strengths = generateStrengths(8)

# RR = RoundRobin(strengths, 64)
# runTournament(RR, graphStep=180)

# SE2 = SingleElimination2(strengths)
# runTournament(SE2)

# DE2 = DoubleElimination2(strengths)
# runTournament(DE2)

# SW = Swiss(strengths)
# runTournament(SW)

# IS = InsertionSort(strengths, bestOf=7)
# runTournament(IS)

# BIS = BinaryInsertionSort(strengths, bestOf=333)
# runTournament(BIS)

# BS = BubbleSort(strengths, bestOf=99)
# runTournament(BS)

# SS = SelectionSort(strengths, bestOf=3)
# runTournament(SS)

# QS = QuickSort(strengths, bestOf=333)
# runTournament(QS)

# MS = MergeSort(strengths, bestOf=3)
# runTournament(MS)

HS = HeapSort(strengths, bestOf=999)
runTournament(HS)