import numpy as np
from typing import List, Dict
import random
import math
from Tournament import Tournament
from utils import *

class MAB(Tournament):
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400)), bestOf=1, verbose=True, explorationFolds=3, patience=3):
        super().__init__(strengths, eloFunc, bestOf, verbose)

        self.explorationFolds = explorationFolds
        self.patience = patience

        self.explorationCount = 0

        self.arms = []
        for i in range(self.numPlayers):
            for j in range(i+1, self.numPlayers):
                self.arms.append((i,j))

        self.numWins   = {x: 0 for x in self.arms}
        self.numLosses = {x: 0 for x in self.arms}

    def getMeanRewards(self) -> Dict[tuple, float]:
        meanRewards = {}
        for match in self.arms:
            wins   = self.numWins[match]
            losses = self.numLosses[match]
            if wins + losses == 0:
                meanRewards[match] = 0.5
            else:
                meanRewards[match] = wins/(wins+losses)
        return meanRewards

    def getNextMatch(self) -> List[int]:
        if len(self.schedule) < len(self.arms)*self.explorationFolds:
            match = self.arms[self.explorationCount]
            self.explorationCount = (self.explorationCount + 1) % len(self.arms)
            return match
        else:
            # Check if a definite ranking has been found AND more than half the arms have been locked-in
            if len(self.arms) <= 0.25*self.numPlayers*(self.numPlayers-1): # and [] not in self.getRanking()
                return None

            # Check if we should 'lock-in' any arms
            prevMatches = [sorted(x) for x in self.schedule[-self.patience:]]
            lockIn = True
            for match in prevMatches[1:]:
                if match != prevMatches[0]:
                    lockIn = False
                    break
            if lockIn:
                lockedInMatch = tuple(prevMatches[0])
                self.arms.remove(lockedInMatch)
                self.numWins.pop(lockedInMatch)
                self.numLosses.pop(lockedInMatch)

            if len(self.arms) == 0:
                return None

            # Now apply whichever particular MAB policy we want to use in order to pick the next arm (match)
            return self.chooseArm()

    def chooseArm(self) -> List[int]:
        return random.choice(self.arms)

    def getRanking(self) -> List[List[int]]:
        return getDominationDegreeRanking(self.winRatesLaplace)[0]

class UCB(MAB):
    def getBoundSizes(self) -> Dict[tuple, float]:
        boundSizes = {}
        for match in self.arms:
            wins   = self.numWins[match]
            losses = self.numLosses[match]
            if wins + losses == 0:
                boundSizes[match] = 0.5
            else:
                boundSizes[match] = math.sqrt((3*math.log(len(self.schedule)))/(2*(self.numWins[match]+self.numLosses[match])))
        return boundSizes

    def getLowerBounds(self) -> Dict[tuple, float]:
        meanRewards = self.getMeanRewards()
        boundSizes  = self.getBoundSizes()
        return {x: meanRewards[x] - boundSizes[x] for x in self.arms}

    def getUpperBounds(self) -> Dict[tuple, float]:
        meanRewards = self.getMeanRewards()
        boundSizes  = self.getBoundSizes()
        return {x: meanRewards[x] + boundSizes[x] for x in self.arms}

    def chooseArm(self) -> List[int]:
        UCBs = self.getUpperBounds()
        LCBs = self.getLowerBounds()

        minDistTo0or1 = 1
        potentialArms = []

        for match in self.arms:
            dist = min(1-UCBs[match], LCBs[match])
            if dist < minDistTo0or1:
                minDistTo0or1 == dist
                potentialArms = [match]
            elif dist == minDistTo0or1:
                potentialArms.append(match)
        
        return random.choice(potentialArms)


class TS(MAB):
    def chooseArm(self) -> List[int]:
        # print(len(self.schedule), len(self.arms))
        maxStrengthExpectation = 0
        nextMatch = []
        for match in self.arms:
            strengthExpectation = np.random.beta(self.numWins[match] + 1, self.numLosses[match] + 1)
            if strengthExpectation > maxStrengthExpectation:
                maxStrengthExpectation = strengthExpectation
                nextMatch = match 
        return nextMatch

class EG(MAB):
    def __init__(self, strengths, eloFunc=lambda x: 1/(1+10**(x/400)), bestOf=1, verbose=True, explorationFolds=3, patience=3, epsilon=0.25):
        super().__init__(strengths, eloFunc, bestOf, verbose, explorationFolds, patience)
        self.epsilon = epsilon

    def chooseArm(self) -> List[int]:
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        else:
            minDistTo0or1 = 1
            potentialNextMatches = []
            for match in self.arms:
                dist = min(1-self.winRatesLaplace[match], self.winRatesLaplace[match])
                if dist < minDistTo0or1:
                    minDistTo0or1 = dist
                    potentialNextMatches = [match]
                elif dist == minDistTo0or1:
                    potentialNextMatches.append(match)
            
            return random.choice(potentialNextMatches)


        

