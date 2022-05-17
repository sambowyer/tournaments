import pandas as pd 
import numpy as np 
from typing import List
from itertools import product
import math

def fixDERounds(rounds: int, matches : int, numPlayers : int) -> int:
    if matches == 2*numPlayers - 2:
        return int(2*math.log2(numPlayers) + 1)
    elif matches == 2*numPlayers - 1:
        return int(2*math.log2(numPlayers) + 2)
    else:
        print("wtf", rounds, matches, numPlayers)
        input()

def averageOutCSV(inputCSV : str, outputCSV : str, identifyingHeaders : List, headersToAverageOut : [], sortOutPositions=True, blankColumnIndex=-1, fixRounds=True, excludeFunc=None):
    '''This function averages out stats from identical runs in a CSV and adds in standard deviation columns for those averaged-out columns too.
    It will also automatically do the same for columns "correctPositions" and "eloCorrectPositions" (if they are present) due to their unique formatting.'''

    tp = pd.read_csv(inputCSV, iterator=True, chunksize=1000)
    df = pd.concat(tp, ignore_index=True)

    df.drop(columns=[df.columns[blankColumnIndex]], inplace=True)

    # Fix the round numbers
    if fixRounds:
        for i in range(len(df)):
            if df.loc[i, "tournament"] == "DE":
                df.loc[i, "numRounds"] = fixDERounds(df.loc[i, "numRounds"], df.loc[i, "numMatches"], df.loc[i, "numPlayers"])
            elif df.loc[i, "tournament"][:2] in ["QS", "MS"]:
                df.loc[i, "numRounds"] = df.loc[i, "numMatches"]
            elif df.loc[i, "tournament"][:2] in ["UC", "TS", "EG"]:
                exp = 3
                if "explorationFolds" in df.columns:
                    exp = df.loc[i, "explorationFolds"]
                df.loc[i, "numRounds"] = df.loc[i, "numMatches"] - exp*df.loc[i, "numPlayers"]

    if excludeFunc is not None:
        print("Mean number of matches:")
        print({x: df[df["tournament"] == x]["numMatches"].mean() for x in df["tournament"].unique()})
        for t in df["tournament"].unique():
            x = df[df["tournament"] == t]
            print(t, {y: x[x["numPlayers"] == y]["numMatches"].mean() for y in df["numPlayers"].unique()})

        # print({x: {y: df[df[df["tournament"] == x]["numMatches"] == y]["numMatches"].mean() for y in df["numPlayers"].unique()} for x in df["tournament"].unique()})
        toDrop = []
        # prevLens = {x: 0 for x in df["tournament"].unique()}
        prevLens = {x: len(df[df["tournament"] == x]) for x in df["tournament"].unique()}
        print("All rows:")
        print(prevLens)
        for t in df["tournament"].unique():
            x = df[df["tournament"] == t]
            print(t, {y: len(x[x["numPlayers"] == y]) for y in df["numPlayers"].unique()})

        # print({x: {y: len(df[df[df["tournament"] == x]["numMatches"] == y]) for y in df["numPlayers"].unique()} for x in df["tournament"].unique()})
        for i in range(len(df)):
            row = df.loc[i]
            if excludeFunc(row):
                # print(row[""])
                toDrop.append(i)
        
        df.drop(toDrop, axis=0, inplace=True)

        newLens = {x: len(df[df["tournament"] == x]) for x in df["tournament"].unique()}

        
        print("With excluded rows removed:")
        print(newLens)
        for t in df["tournament"].unique():
            x = df[df["tournament"] == t]
            print(t, {y: len(x[x["numPlayers"] == y]) for y in df["numPlayers"].unique()})
        # print({x: {y: len(df[df[df["tournament"] == x]["numMatches"] == y]) for y in df["numPlayers"].unique()} for x in df["tournament"].unique()})

        print("Mean number of matches:")
        print({x: df[df["tournament"] == x]["numMatches"].mean() for x in df["tournament"].unique()})
        for t in df["tournament"].unique():
            x = df[df["tournament"] == t]
            print(t, {y: x[x["numPlayers"] == y]["numMatches"].mean() for y in df["numPlayers"].unique()})

        # print({x: {y: df[df[df["tournament"] == x]["numMatches"] == y]["numMatches"].mean() for y in df["numPlayers"].unique()} for x in df["tournament"].unique()})
        


    # input()
    # DE = df[df["tournament"] == "DE"]
    # df = df[df["tournament"] != "DE"]
    # DE = DE[DE["cosine0"] > 0.01]
    # df = pd.concat([df,DE])

    dataFinal = {header : [] for header in df.columns}
    
    for header in headersToAverageOut:
        dataFinal[f"{header}STD"] = []
    if sortOutPositions:
        dataFinal["correctPositionsSTD"] = []
        dataFinal["eloCorrectPositionsSTD"] = []

    identifyingValues = {header : df[header].unique() for header in identifyingHeaders}

    for combination in product(*[df[header].unique() for header in identifyingHeaders]):
        filtered = df.copy()

        for i, header in enumerate(identifyingValues):
            filtered = filtered[filtered[header] == combination[i]]

        if len(filtered) != 0:

            # Sort out the columns to be average out (and add their standard deviations too)
            rowValues = filtered.values.tolist()[0]
            row = {header : rowValues[i] for i, header in enumerate(filtered.columns)}
            
            for header in headersToAverageOut:
                row[header] = filtered[header].mean()
                row[f"{header}STD"] = filtered[header].std()

            # And sort out the "correctPositions" and "eloCorrectPositions" columns too
            if sortOutPositions:
                for header in ["correctPositions", "eloCorrectPositions"]:
                    if header in filtered:
                        positions = np.asarray([[float(y) for y in x.split("_")] for x in filtered[header]])

                        means = positions.mean(axis=0)
                        stds  = positions.std(axis=0)

                        means = str(list(means)).strip("[]").replace(", ", "_")
                        stds  = str(list(stds)).strip("[]").replace(", ", "_")

                        # print(row)

                        row[header] = means
                        row[f"{header}STD"] = stds

            for header in row:
                dataFinal[header].append(row[header])


    dfFinal = pd.DataFrame(dataFinal)

    print(dfFinal)

    dfFinal.to_csv(outputCSV, index=False, header=True)


# averageOutCSV("csvs/classicalAndSortingTests.csv", "csvs/classicalAndSortingTestsAveraged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])

# averageOutCSV("csvs/classicalAndSortingTestsFIXED2.csv", "csvs/classicalAndSortingTestsAveraged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
# averageOutCSV("csvs/MABTuningTestsFIXED.csv", "csvs/MABTuningTestsAveraged2.csv", ["tournament","explorationFolds","patience","maxLockInProportion", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], sortOutPositions=False)
# averageOutCSV("csvs/MABMainTests.csv", "csvs/MABMainTestsAveraged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
# averageOutCSV("csvs/transitivityTests.csv", "csvs/transitivityTestsAveraged.csv", ["tournament", "numPlayers", "strongTransitivity"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
# averageOutCSV("csvs/fixedMatchesTests12000.csv", "csvs/fixedMatchesTests12000Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], fixRounds=False)
# averageOutCSV("csvs/fixedMatchesTests2400.csv", "csvs/fixedMatchesTests2400Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], fixRounds=False)
# averageOutCSV("csvs/fixedMatchesTests120.csv", "csvs/fixedMatchesTests120Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], fixRounds=False)


# averageOutCSV("csvs/MABTuningTestsAveraged2.csv", "csvs/MABTuningTestsAveraged2_explorationFolds.csv", ["tournament","explorationFolds", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], sortOutPositions=False)
# averageOutCSV("csvs/MABTuningTestsAveraged2.csv", "csvs/MABTuningTestsAveraged2_patience.csv", ["tournament","patience", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], sortOutPositions=False)
# averageOutCSV("csvs/MABTuningTestsAveraged2.csv", "csvs/MABTuningTestsAveraged2_maxLockInProportion.csv", ["tournament","maxLockInProportion", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], sortOutPositions=False)
# averageOutCSV("csvs/MABTuningTestsAveraged2_epsilon.csv", "csvs/MABTuningTestsAveraged2_epsilonAveraged.csv", ["tournament", "epsilon", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], sortOutPositions=False, blankColumnIndex=-2)

# averageOutCSV("csvs/MABMainTests.csv", "csvs/MABMainTestsAveraged_NO_HORIZON.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], excludeFunc=lambda x: x["numMatches"] > 50000)# 100*x["numPlayers"]*(x["numPlayers"]-1))

# averageOutCSV("csvs/MABMainTestsLowExp.csv", "csvs/MABMainTestsLowExpAveraged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
# averageOutCSV("csvs/MABMainTestsLowExp.csv", "csvs/MABMainTestsLowExpNoTS32Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], excludeFunc=lambda x: x["numMatches"] >= 100*x["numPlayers"]*(x["numPlayers"]-1) and x["tournament"] == "TS")
# averageOutCSV("csvs/transitivityTestsLowExp.csv", "csvs/transitivityTestsLowExpAveraged.csv", ["tournament", "numPlayers", "strongTransitivity"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
# averageOutCSV("csvs/fixedMatchesTestsLowExp12000.csv", "csvs/fixedMatchesTestsLowExp12000Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], fixRounds=False)
# averageOutCSV("csvs/fixedMatchesTestsLowExp2400.csv", "csvs/fixedMatchesTestsLowExp2400Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], fixRounds=False)
# averageOutCSV("csvs/fixedMatchesTestsLowExp120.csv", "csvs/fixedMatchesTestsLowExp120Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], fixRounds=False)


averageOutCSV("csvs/RRDomTest1000.csv", "csvs/RRDomTest1000Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
