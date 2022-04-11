import numpy as np
from typing import List
import random

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

def cosineDistance(vec1 : List[int], vec2 : List[int], start=1) -> float:
    '''Returns the cosine distance between two vectors'''
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    m  = min(v1)
    v1 += start-m
    v2 += start-m
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def getPositionsVector(ranking : List[int]) -> List[int]:
    '''Takes a 'ranking' of players in order from winner to loser and returns a vector whose ith element is the position of the ith player in the ranking.'''
    return [ranking.index(i) for i in range(len(ranking))]

def getRankingSimilarity(ranking1 : List[int], ranking2 : List[int]) -> float:
    return cosineDistance(getPositionsVector(ranking1), getPositionsVector(ranking2))
