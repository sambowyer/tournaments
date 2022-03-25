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
    def __init__(self, strengths : np.ndarray, eloFunc=lambda x: 1/(1+10**(x/400))):
        self.strengths = strengths 
        self.eloFunc   = eloFunc

        self.isFinished = False

        self.numPlayers  = strengths.shape[0]
        self.schedule    = []  # records all the matches that have been played up to this point
        self.resultsList = []  # records all the results (each 0 or 1) of all matches that have already been played

        self.resultsMatrix   = np.zeros([self.numPlayers, self.numPlayers])
        self.eloScores       = [1000 for i in range(self.numPlayers)]
        self.winRates        = np.zeros([self.numPlayers, self.numPlayers])
        self.winRatesLaplace = np.array([[0 if x==y else 0.5 for x in range(self.numPlayers)] for y in range(self.numPlayers)])

    def getNextMatch(self) -> List[int]:
        '''Should return a size-2 list containing the two competitors (i.e. 2 ints) playing the next match. If no more matches in schedule, then return None.
        This is perhaps the most important way to distinguish between different tournament systems.'''
        return [0,1]

    def runNextMatch(self):
        '''Runs the next match, updates the schedule, resultsList, resultsMatrix, eloScores, winRates, winRatesLaplace.'''
        match = self.getNextMatch()
        self.schedule.append(match)
        
        if match is None:
            self.isFinished = True
        else:
            outcome = random.random()

            # We set result to 0 if match[0] wins, and 1 if match[1] wins
            result = 0 if outcome <= self.strengths[match[0], match[1]] else 1
            self.resultsList.append(result)

            # These might help to follow along with the rest of this method
            winner = match[result]
            loser  = match[1-result]

            print(f"{winner} beat {loser}")
            
            # Now to update the state of the tournament
            # First update the results matrix
            self.resultsMatrix[winner, loser] += 1

            # Then the Elo scores
            eloDifference      = self.eloScores[winner] - self.eloScores[loser]
            pointsTransfer     = self.eloFunc(eloDifference)
            self.eloScores[winner] += 25*pointsTransfer
            self.eloScores[loser]  -= 25*pointsTransfer

            # Finally the win rates matrix and the win rates matrix with Laplace succession
            self.winRates[winner, loser] = self.resultsMatrix[winner, loser] / (self.resultsMatrix[winner, loser] + self.resultsMatrix[loser, winner])
            self.winRates[loser, winner] = self.resultsMatrix[loser, winner] / (self.resultsMatrix[winner, loser] + self.resultsMatrix[loser, winner])

            self.winRatesLaplace[winner, loser] = (1 + self.resultsMatrix[winner, loser]) / (2 + self.resultsMatrix[winner, loser] + self.resultsMatrix[loser, winner])
            self.winRatesLaplace[loser, winner] = (1 + self.resultsMatrix[loser, winner]) / (2 + self.resultsMatrix[winner, loser] + self.resultsMatrix[loser, winner])

    def getTotalWins(self) -> List[int]:
        '''Returns a list of the total number of wins that each player has achieved.'''
        return [sum(x) for x in self.resultsMatrix]
    
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
    def __init__(self, strengths, numFolds, eloFunc=lambda x: 1/(1+10**(x/400))):
        super().__init__(strengths, eloFunc)
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
            return self.getTotalWinRanking()
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
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400))):
        super().__init__(strengths, eloFunc)
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
            return self.getTotalWinRanking()
        else:
            print("Not finished yet.")  #TODO: change this

class DoubleElimination2(Tournament):
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400))):
        super().__init__(strengths, eloFunc)
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
            return self.getTotalWinRanking()
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

class SortingAlgorithm(Tournament):
    pass

def runTournament(tournament : Tournament, graphStep=1):
    totalWinsHistory = []
    eloScoresHistory = []

    while not tournament.isFinished:
        totalWinsHistory.append(tournament.getTotalWins())
        eloScoresHistory.append(tournament.eloScores.copy())

        tournament.runNextMatch()

    # TODO: Clean up this monstrosity (USE PANDAS DATAFRAMES)
    strPlayers, strNumbers   = getAverageStrengthRanking(tournament.strengths)
    winPlayers, winNumbers   = tournament.getTotalWinRanking()
    eloPlayers, eloNumbers   = tournament.getEloRanking()
    awrPlayers, awrNumbers   = tournament.getAverageWinRateRanking()
    awrlPlayers, awrlNumbers = tournament.getAverageWinRateLaplaceRanking()

    trueRanking = getTrueRanking(tournament.strengths)

    print("Rank   True   Player (Str)   Player (Wins)   Player (Elos)   Player (AWR)   Player (AWRL)")
    print("-----------------------------------------------------------------------------------------")
    for i in range(tournament.numPlayers):
        print(f" {i+1:2}     {trueRanking[i]:2}     {strPlayers[i]:2} ({strNumbers[i]:.2f})      {winPlayers[i]:2} ({winNumbers[i]})       {eloPlayers[i]:2} ({eloNumbers[i]:7.2f})    {awrPlayers[i]:2} ({awrNumbers[i]:.2f})      {awrlPlayers[i]:2} ({awrlNumbers[i]:.2f})")
        
    print(f"Cosine similarities to true ranking:\nStrs: {getRankingSimilarity(trueRanking, strPlayers)}\nWins: {getRankingSimilarity(trueRanking, winPlayers)}\nElos: {getRankingSimilarity(trueRanking, eloPlayers)}")

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

# SE = SingleElimination(strengths)
# runTournament(SE)

# SE2 = SingleElimination2(strengths)
# runTournament(SE2)

DE2 = DoubleElimination2(strengths)
runTournament(DE2)

print(getRankingSimilarity([0,1,2,3,4,5,6,7], [7,6,5,4,3,2,1,0]))