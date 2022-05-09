import pandas as pd 
import numpy as np 
from typing import List
from itertools import product

def averageOutCSV(inputCSV : str, outputCSV : str, identifyingHeaders : List, headersToAverageOut : [], sortOutPositions=True, blankColumnIndex=-1):
    '''This function averages out stats from identical runs in a CSV and adds in standard deviation columns for those averaged-out columns too.
    It will also automatically do the same for columns "correctPositions" and "eloCorrectPositions" (if they are present) due to their unique formatting.'''

    tp = pd.read_csv(inputCSV, iterator=True, chunksize=1000)
    df = pd.concat(tp, ignore_index=True)

    df.drop(columns=[df.columns[blankColumnIndex]], inplace=True)

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


averageOutCSV("csvs/classicalAndSortingTests.csv", "csvs/classicalAndSortingTestsAveraged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
# averageOutCSV("csvs/MABTuningTestsFIXED.csv", "csvs/MABTuningTestsAveraged2.csv", ["tournament","explorationFolds","patience","maxLockInProportion", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], sortOutPositions=False)
# averageOutCSV("csvs/MABMainTests.csv", "csvs/MABMainTestsAveraged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
# averageOutCSV("csvs/transitivityTests.csv", "csvs/transitivityTestsAveraged.csv", ["tournament", "numPlayers", "strongTransitivity"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
# averageOutCSV("csvs/fixedMatchesTests12000.csv", "csvs/fixedMatchesTests12000Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
# averageOutCSV("csvs/fixedMatchesTests2400.csv", "csvs/fixedMatchesTests2400Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])
# averageOutCSV("csvs/fixedMatchesTests120.csv", "csvs/fixedMatchesTests120Averaged.csv", ["tournament", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])


# averageOutCSV("csvs/MABTuningTestsAveraged2.csv", "csvs/MABTuningTestsAveraged2_explorationFolds.csv", ["tournament","explorationFolds", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], sortOutPositions=False)
# averageOutCSV("csvs/MABTuningTestsAveraged2.csv", "csvs/MABTuningTestsAveraged2_patience.csv", ["tournament","patience", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], sortOutPositions=False)
# averageOutCSV("csvs/MABTuningTestsAveraged2.csv", "csvs/MABTuningTestsAveraged2_maxLockInProportion.csv", ["tournament","maxLockInProportion", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], sortOutPositions=False)
# averageOutCSV("csvs/MABTuningTestsAveraged2_epsilon.csv", "csvs/MABTuningTestsAveraged2_epsilonAveraged.csv", ["tournament", "epsilon", "numPlayers"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"], sortOutPositions=False, blankColumnIndex=-2)

