import numpy as np
from typing import List
import random
import math
from Tournament import Tournament
from utils import *


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
        # print(self.ranking)
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

        # print(arr)
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
