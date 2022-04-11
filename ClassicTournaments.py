import numpy as np
from typing import List
import random
import math
from collections import deque
from Tournament import Tournament
from utils import *

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

class SingleElimination(Tournament):
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400)), bestOf=1):
        super().__init__(strengths, eloFunc, bestOf)
        self.currentRound = SingleEliminationRound(strengths, list(range(self.numPlayers)), eloFunc)

    def getNextMatch(self) -> List[int]:
        nextMatch = self.currentRound.getNextMatch()
        if nextMatch is None:  # Round has finished
            winners = self.currentRound.getWinners(self.resultsList[-self.currentRound.matchNo:])
            
            if len(winners) == 1:  # Tournament has finished
                return None
                
            nextRoundStrengths = getStrengthsSubmatrix(self.strengths, winners)

            self.currentRound = SingleEliminationRound(nextRoundStrengths, winners, self.eloFunc)
        
            return self.getNextMatch()

        return nextMatch

    def getRanking(self) -> (List[int], List[int]):
        if self.isFinished:
            return self.getTotalWinRanking()[0]
        else:
            print("Not finished yet.")  #TODO: change this

class DoubleElimination(Tournament):
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
                    
                nextRoundStrengths = getStrengthsSubmatrix(self.strengths, winners)
                self.currentWinnerRound = SingleEliminationRound(nextRoundStrengths, winners, self.eloFunc)
                self.currentTree = "loser"

                # with open("aa.txt", "a") as f:
                #     self.wc += 1
                #     f.write(f"wc {self.wc} {len(self.resultsList)}\n")

                return self.getNextMatch()

            return nextMatch

        elif self.currentTree == "loser":
        
            if self.isLoserOnlyRound:
                if self.currentLoserRound is None:
                    nextRoundStrengths = getStrengthsSubmatrix(self.strengths, self.newLosers)
                    self.currentLoserRound = SingleEliminationRound(nextRoundStrengths, self.newLosers)

                nextMatch = self.currentLoserRound.getNextMatch()
                if nextMatch is None:  # Round has finished

                    self.successfulLosers = self.currentLoserRound.getWinners(self.resultsList[-self.currentLoserRound.matchNo:])
                    self.currentTree = "winner"
                    self.isLoserOnlyRound = False

                    self.currentLoserRound = None

                    # with open("aa.txt", "a") as f:
                    #     self.lc += 1
                    #     f.write(f"lc {self.lc} {len(self.resultsList)}\n")

                    return self.getNextMatch()

                return nextMatch

            else:
                if self.currentLoserRound is None:
                    
                    nextRoundStrengths = getStrengthsSubmatrix(self.strengths, self.newLosers+self.successfulLosers)
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
                    nextRoundStrengths = getStrengthsSubmatrix(self.strengths, self.successfulLosers)
                    self.currentLoserRound = SingleEliminationRound(nextRoundStrengths, self.successfulLosers)

                    # with open("aa.txt", "a") as f:
                    #     self.lc += 1
                    #     f.write(f"lc {self.lc} {len(self.resultsList)}\n")

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
