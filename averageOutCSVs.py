import pandas as pd 
import numpy as np 
from typing import List
from itertools import product

def averageOutCSV(inputCSV : str, outputCSV : str, identifyingHeaders : List, headersToAverageOut : []):
    '''This function averages out stats from identical runs in a CSV and adds in standard deviation columns for those averaged-out columns too.
    It will also automatically do the same for columns "correctPositions" and "eloCorrectPositions" (if they are present) due to their unique formatting.'''
    
    tp = pd.read_csv(inputCSV, iterator=True, chunksize=1000)
    df = pd.concat(tp, ignore_index=True)

    df.drop(columns=[df.columns[-1]], inplace=True)

    dataFinal = {header : [] for header in df.columns}
    for header in headersToAverageOut + ["correctPositions", "eloCorrectPositions"]:
        dataFinal[f"{header}STD"] = []

    identifyingValues = {header : df[header].unique() for header in identifyingHeaders}

    for combination in product(*[df[header].unique() for header in identifyingHeaders]):
        filtered = df.copy()

        for i, header in enumerate(identifyingValues):
            filtered = filtered[filtered[header] == combination[i]]

        # Sort out the columns to be average out (and add their standard deviations too)
        rowValues = filtered.values.tolist()[0]
        row = {header : rowValues[i] for i, header in enumerate(filtered.columns)}
        
        for header in headersToAverageOut:
            row[header] = filtered[header].mean()
            row[f"{header}STD"] = filtered[header].mean()

        # And sort out the "correctPositions" and "eloCorrectPositions" columns too
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
averageOutCSV("csvs/MABTuningTests.csv", "csvs/MABTuningTestsAveraged.csv", ["tournament","numPlayers","explorationFolds","patience","maxLockInProportion"], ["numMatches", "numRounds", "cosine0", "cosine1", "eloCosine0", "eloCosine1"])






