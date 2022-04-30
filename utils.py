import numpy as np
from typing import List, Generator
import random
from itertools import permutations
import math

def generateStrengths(numPlayers : int, strongTransitivity = False) -> np.ndarray:
    '''Returns a square numpy array of relative strengths between players.'''

    strengths = np.zeros([numPlayers,numPlayers])
    tempStrength = 0
    for i in range(numPlayers):
        for j in range(i+1, numPlayers):
            transitive = False
            while not transitive:
                transitive = True
                tempStrength = random.uniform(0,1)
                if tempStrength == 0.5:
                    transitive = False
                else:
                    for k in range(numPlayers):
                        if k not in [i,j] and strengths[i,k] > 0 and strengths[k,j] > 0:
                            if not strongTransitivity:
                                if strengths[i,k] > 0.5 and strengths[k,j] > 0.5 and tempStrength < 0.5:
                                    transitive = False
                                elif strengths[i,k] < 0.5 and strengths[k,j] < 0.5 and tempStrength > 0.5:
                                    transitive = False
                            else:
                                if strengths[i,k] > 0.5 and strengths[k,j] > 0.5 and tempStrength < max(strengths[i,k], strengths[k,j]):
                                    transitive = False
                                elif strengths[i,k] < 0.5 and strengths[k,j] < 0.5 and tempStrength > min(strengths[i,k], strengths[k,j]):
                                    transitive = False
            strengths[i,j] = tempStrength
            strengths[j,i] = 1-tempStrength

    return strengths

def getStrengthsSubmatrix(strengths: np.ndarray, validPlayers : List[int]) -> np.ndarray:
    resultsTemp = [i in validPlayers for i in range(strengths.shape[0])]  # so True if i won, false if i lost
    nextRoundStrengths = np.compress(resultsTemp, strengths, axis=0)
    nextRoundStrengths = np.compress(resultsTemp, nextRoundStrengths, axis=1)
    return nextRoundStrengths

def combineJointPositionsInRanking(ranking : List[int], values : List) -> (List[List[int]], List):
        outputPlayers = [[ranking[0]]]
        outputValues  = [values[0]]
        currentValue = values[0]
        lastNonEmptyIndex=0
        
        for i in range(1,len(ranking)):
            if values[i] == currentValue:
                outputPlayers[lastNonEmptyIndex].append(ranking[i])
                outputPlayers.append([])
                outputValues.append(None)
            else:
                outputPlayers.append([ranking[i]])
                outputValues.append(values[i])
                lastNonEmptyIndex = i
                currentValue = values[i]
        
        return outputPlayers, outputValues

def getDominationDegreeRanking(dominationMatrix  : np.ndarray) -> (List[List[int]], List[int]):
    '''Returns the longest path in the dominance graph given by 'dominationMatrix')'''
    n = dominationMatrix.shape[0]
    outDegrees = [(i, sum([1 if (dominationMatrix[i,j] > 0.5 and not i==j) else 0 for j in range(n)])) for i in range(n)]
    outDegrees = sorted(outDegrees, key = lambda x: x[1], reverse=True)
    players = []
    values  = []
    for x in outDegrees:
        players.append(x[0])
        values.append(x[1])
    
    return combineJointPositionsInRanking(players, values)

def getTrueRanking(strengths : np.ndarray) -> List[int]:
    '''Returns the 'true' ranking of players in 'strengths' (i.e. the longest path in the dominance graph given by 'strengths')'''
    return getDominationDegreeRanking(strengths)[0]

def getAverageStrengthRanking(strengths) -> (List[int], List[float]):
    ranking = range(strengths.shape[0])
    averageStrengths = [sum(x)/(strengths.shape[0]-1) for x in strengths]
    ranking = sorted(ranking, key = lambda x: averageStrengths[x], reverse = True)

    return combineJointPositionsInRanking(ranking, sorted(averageStrengths, reverse=True))

def cosineSimilarity(vec1 : List[int], vec2 : List[int], start=1) -> float:
    '''Returns the cosine distance between two vectors'''
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    m  = min(v1)
    v1 += start-m
    v2 += start-m
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def getPositionsVector(ranking : List[List[int]], startAtZero=True) -> List[int]:
    '''Takes a 'ranking' of players in order from winner to loser and returns a vector whose ith element is the position of the ith player in the ranking.'''
    if startAtZero:
        return [ranking.index([i]) for i in range(len(ranking))]
    else:
        return [ranking.index([i])+1 for i in range(len(ranking))]

def randomlyCollapseJointPositions(ranking : List[List[int]]) -> List[List[int]]:
    '''Given a ranking that may contain joint positions, this will return a ranking that aligns with the original in all places except between players that share the same original joint-position.'''
    out = []
    for x in ranking:
        if len(x) == 1:
            out.append(x)
        elif len(x) > 1:
            jointPlayers = x.copy()
            random.shuffle(jointPlayers)
            for y in jointPlayers:
                out.append([y])
    return out

def getNumberOfPossibleDefiniteRankings(ranking : List[List[int]]) -> int:
    return np.prod([math.factorial(len(x)) for x in ranking])

def getRankingSimilarity(ranking1 : List[List[int]], ranking2 : List[List[int]], numSamples="all") -> (float, float):
    '''If numSamples="all", this returns the average cosine similarity between all possible rankings that could come from 'ranking1' and 'ranking2' if we remove any joint-positions along with the standard deviation of those similarities.
    Otherwise, we pick numSamples (an int) number of possible rankings that would align with each ranking and the corresponding joint-positions and return the mean and UNBIASED sample standard deviation of those samples. '''
    # First check whether either of ranking1 and ranking2 don't actually contain any joint positions (will help cut down on computations down the line)
    noJoints1 = max([len(x) for x in ranking1]) == 1
    noJoints2 = max([len(x) for x in ranking2]) == 1

    # TODO: Here we split up the case that only one of noJoints1 and noJoints2 is true in order to limit calls to randomlyCollapseJointPositions() (which is relatively inexpensive, really), but is this efficiency worth the ugly (semi-redundant) code below?

    if noJoints1 and noJoints2:
        return cosineSimilarity(getPositionsVector(ranking1), getPositionsVector(ranking2)), 0
    elif not noJoints1 and noJoints2:
        return getRankingSimilarity(ranking2, ranking1, numSamples)
    elif noJoints1 and not noJoints2:
        # so we only need to collapse joint positions for ranking2
        if numSamples == "all":
            numPossibleComparisons = getNumberOfPossibleDefiniteRankings(ranking2)
            total = 0
            totalSquares = 0
            
            possibleDefiniteRankings = getPossibleDefiniteRankings(ranking2)
            for definiteRanking2 in possibleDefiniteRankings:
                similarity, _ = getRankingSimilarity(ranking1, definiteRanking2)
                # print(definiteRanking2, similarity)
                total += similarity
                totalSquares += similarity**2
            mean = total/numPossibleComparisons
            return mean, math.sqrt((numPossibleComparisons/(numPossibleComparisons-1))*((totalSquares/numPossibleComparisons)-mean**2))

        else:
            total = 0
            totalSquares = 0
            for _ in range(numSamples):
                similarity, _ = getRankingSimilarity(ranking1, randomlyCollapseJointPositions(ranking2))
                total += similarity
                totalSquares += similarity**2
            mean = total/numSamples
            return mean, math.sqrt((numSamples/(numSamples-1))*((totalSquares/numSamples)-mean**2))
    else:
        # so we need to collapse joint positions for ranking1 and ranking2
        if numSamples == "all":
            numPossibleComparisons = getNumberOfPossibleDefiniteRankings(ranking1) * getNumberOfPossibleDefiniteRankings(ranking2)
            total = 0
            totalSquares = 0
            
            possibleDefiniteRankings1 = getPossibleDefiniteRankings(ranking1)
            for definiteRanking1 in possibleDefiniteRankings1:
                possibleDefiniteRankings2 = getPossibleDefiniteRankings(ranking2)
                for definiteRanking2 in possibleDefiniteRankings2:
                    similarity, _ = getRankingSimilarity(definiteRanking1, definiteRanking2)
                    total += similarity
                    totalSquares += similarity**2
            mean = total / numPossibleComparisons
            return mean, math.sqrt((numPossibleComparisons/(numPossibleComparisons-1))*((totalSquares/numPossibleComparisons)-mean**2))

        else:
            total = 0
            totalSquares = 0
            for _ in range(numSamples):
                similarity, _ = getRankingSimilarity(ranking1, randomlyCollapseJointPositions(ranking2))
                total += similarity
                totalSquares += similarity**2
            mean = total/numSamples
            return mean, math.sqrt((numSamples/(numSamples-1))*((totalSquares/numSamples)-mean**2))

def getPossibleDefiniteRankings(ranking : List[List[int]], maxNum=None) -> Generator[List[List[int]], None, None]:
    if maxNum is None:
        maxNum = getNumberOfPossibleDefiniteRankings(ranking)

    jointPlayers = []
    jointIndices = []
    for i, x in enumerate(ranking):
        if len(x)>1:
            jointPlayers.append(x)
            jointIndices.append(i)

    totalPermNos   = []
    currentPermNos = []
    permGenerators = []
    currentPerms = []
    for x in jointPlayers:
        totalPermNos.append(math.factorial(len(x)))
        currentPermNos.append(0)
        permGenerators.append(None)
        currentPerms.append([])

    for currentNum in range(maxNum):
        for i in range(len(jointPlayers)):
            if currentPermNos[i] == 0:
                permGenerators[i] = permutations(jointPlayers[i])
            currentPermNos[i] += 1
            # print("o", currentPerms[i])
            currentPerms[i] = next(permGenerators[i])
            if currentPermNos[i] == totalPermNos[i]:
                currentPermNos[i] = 0
            elif currentNum != 0:
                break

        # print(currentPermNos, currentPerms)
        # print("aaaaaa")
        
        currentRank = []
        jointPositionsCollapsed = 0
        for x in ranking:
            if len(x) == 1:
                currentRank.append(x)
            elif len(x) > 1:
                for i in range(len(x)):
                    p = currentPerms[jointPositionsCollapsed]
                    currentRank.append([p[i]])
                jointPositionsCollapsed += 1
        
        yield currentRank

def proportionCorrectPositionsVector(correctRanking : List[List[int]], ranking : List[List[int]]) -> List[float]:
    vec = []
    mostRecentJointPlace = []
    for i in range(len(correctRanking)):
        if len(ranking[i]) == 1:
            mostRecentJointPlace = []
            if correctRanking[i] == ranking[i]:
                vec.append(1)
            else:
                vec.append(0)
        else:
            if len(ranking[i]) > 1:
                mostRecentJointPlace = ranking[i]
            
            if correctRanking[i][0] in mostRecentJointPlace:
                vec.append(1/len(mostRecentJointPlace))
            else:
                vec.append(0)
                
    return np.asarray(vec)
            
def combineRankings(rankings : List[List[List[int]]]) -> List[List[int]]:
    '''Combines multiple rankings into a single ranking based on a weighted-sum of positions for each player'''
    # For each ranking we assign 0 points for winning, 1 point for second place etc. and then create our final ranking on these points
    totalPoints = [0 for i in range(len(rankings[0]))]
    for ranking in rankings:
        for points, position in enumerate(ranking):
            for player in position:
                totalPoints[player] += points

    newRanking = [i for i in range(len(rankings[0]))]
    newRanking.sort(key=lambda x: totalPoints[x])

    # Now go through newRanking and combine any adjacent players who have the same number of points
    actualNewRanking = []
    currentPlayers = [newRanking[0]]
    for player in newRanking[1:]:
        if len(currentPlayers) == 0 or totalPoints[player] == totalPoints[currentPlayers[0]]:
            currentPlayers.append(player)
        else:
            actualNewRanking.append(currentPlayers)
            for _ in range(len(currentPlayers)-1):
                actualNewRanking.append([])
            currentPlayers = [player]

    actualNewRanking.append(currentPlayers)
    for _ in range(len(currentPlayers)-1):
        actualNewRanking.append([])

    return actualNewRanking


if __name__ == '__main__':

    rank = [[0],[1,2],[],[4],[3,5,6],[],[],[7,8],[],[9],[10]]
    # rank = [[0,1,2],[],[3],[4,5],[]]
    # rank = [[0,1,2],[],[3],[4,5],[], [6,7],[]]
    print(rank)
    g = getPossibleDefiniteRankings(rank)
    i=0
    rs = []
    for r in g:
        i+=1
        print(r)
        rs.append(r)
    print(i, getNumberOfPossibleDefiniteRankings(rank))

    for i, r in enumerate(rs):
        if r in rs[i+1:]:
            print("oops")

    print(getRankingSimilarity(rank, [[i] for i in range(11)]))
    print(proportionCorrectPositionsVector([[i] for i in range(11)], rank))

    rank1 = [[0],[1],[2],[3],[4],[5],[6]]
    rank2 = [[0],[2],[1],[3],[5],[4],[6]]
    rank3 = [[0],[1],[2],[3],[4,5,6],[],[]]
    ranks = [rank1, rank2, rank3]

    print(combineRankings(ranks))