import numpy as np 
import random
import math
import matplotlib.pyplot as plt

class ScoreKeeper:
    def __init__(self, numPlayers, eloFunc):
        self.results = np.zeros([numPlayers, numPlayers])
        self.elos    = [1000.0]*numPlayers #np.full(numPlayers, 1000.0)
        self.eloFunc = eloFunc

    def update(self, match, result):
        self.updateSimple(match, result)
        self.updateElo(match, result)

    def updateSimple(self, match, result):
        self.results[match[0], match[1]] += result
        self.results[match[1], match[0]] += 1-result

    def updateElo(self, match, result):
        eloDifference  = self.elos[match[result]] - self.elos[match[1-result]]
        pointsTransfer = self.eloFunc(eloDifference)
        # print(pointsTransfer)
        self.elos[match[1-result]] += 25*pointsTransfer
        self.elos[match[result]] -= 25*pointsTransfer

    def updateColleys(self, match, result):
        # TODO: implement Colley's method
        pass

class Scheduler:
    def __init__(self, strengths):
        self.strengths = strengths
        self.schedule = []

    def generateAllMatches(self):
        pass 

    def getNextMatch(self, scoreKeeper):
        pass

class Tournament:
    def __init__(self, strengths : np.ndarray, scheduler : Scheduler, eloFunc = lambda x: 1/(1+10**(x/400))):
        self.scoreKeeper = ScoreKeeper(strengths.shape[0], eloFunc)
        self.numPlayers  = strengths.shape[0]
        self.strengths   = strengths 
        self.scheduler   = scheduler
        self.scoreKeeper = ScoreKeeper(strengths.shape[0], eloFunc)
        self.finished    = False

    def runNextMatch(self):
        pass

    def run(self):
        while not self.finished:
            self.runNextMatch()

    def getResults(self):
        return self.scoreKeeper.results

    def getElos(self):
        return self.scoreKeeper.elos

def generateStrengths(numPlayers : int) -> np.ndarray:
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

### Round Robin 
class RRScheduler(Scheduler):
    def __init__(self, strengths, numFolds):
        super().__init__(strengths)
        self.numFolds = numFolds
        self.generateAllMatches
        self.currentMatchNum = 0
        
        self.generateAllMatches()

    def generateAllMatches(self):
        players = np.arange(strengths.shape[0])
        # singleFold = [np.transpose([np.tile(players, len(players)), np.repeat(players, len(players))])] 
        # singleFold = singleFold[singleFold != singleFold]  # remove matches where x plays x
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                self.schedule.append([i,j])

        # self.schedule = np.repeat(self.schedule, self.numFolds)
        self.schedule *= self.numFolds
        # print(self.schedule)

    def getNextMatch(self, scoreKeeper):
        if self.currentMatchNum < len(self.schedule):
            match = self.schedule[self.currentMatchNum]
            self.currentMatchNum += 1
            return match 
        else:
            return None


class RR(Tournament):
    def __init__(self, strengths : np.ndarray, numFolds : int):
        scheduler = RRScheduler(strengths, numFolds)
        # print(scheduler.schedule)
        super().__init__(strengths, scheduler)

strengths = (generateStrengths(10))
rr = RR(strengths, 10)
winsHistory = [[0]*10]
elosHistory = [[1000]*10]
for round in range(10):
    for game in range(45):
        rr.runNextMatch()
    results = rr.getResults()
    # print(results)
    wins = [sum(x) for x in results]
    elos = rr.getElos()
    print(elos[0] == elosHistory[round-1][0])
    if (round+1) % 25 == 0:
        print("player  wins   elo")
        for x in range(len(wins)):
            print(f"{x}:      {wins[x]}   {elos[x]:.2f}")
    winsHistory.append(wins)
    elosHistory.append(elos)

print(rr.getResults())
print(elosHistory)
gameNums = np.arange(11)

ax =plt.subplot(1,2,1)
for player in range(10):
    plt.plot(gameNums, [x[player] for x in winsHistory])

ax.set(xlabel='Game #', ylabel='Number of wins')

ax =plt.subplot(1,2,2)
for player in range(10):
    plt.plot(gameNums, [x[player] for x in elosHistory])

ax.set(xlabel='Game #', ylabel='Elo rating')

plt.show()


