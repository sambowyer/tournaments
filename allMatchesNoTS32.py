from analyse import *
import pandas as pd 

classicalNonRRDF = pd.read_csv("csvs/classicalAndSortingTestsAveraged.csv")
classicalNonRRDF = filterDF(classicalNonRRDF, "tournament", ["SE", "SE3pp", "DE", "SW"])


classicalAndSortingDF = pd.read_csv("csvs/classicalAndSortingTestsAveraged.csv")
classicalNames = [f"RR{x}" for x in [1,2,3,5,10,25,50,100]] + ["SE", "SE3pp", "DE", "SW"]
sortingAlgos   = ["IS", "BIS", "BS", "SS", "QS", "MS", "HS"]
bestOf1SortingNames = [algo+"1" for algo in sortingAlgos]
sortingNames = bestOf1SortingNames.copy()
for bestOf in [1,3,5,7,9,51,101]:
    sortingNames += [algo+str(bestOf) for algo in sortingAlgos]

classicalDF = filterDF(classicalAndSortingDF, "tournament", classicalNames)
sortingDF   = filterDF(classicalAndSortingDF, "tournament", bestOf1SortingNames)
sortingDF["tournament"] = [name[:2] if name[:3] != "BIS" else "BIS" for name in sortingDF["tournament"].values]


MABMainDF = pd.read_csv("csvs/MABMainTestsLowExpAveraged.csv")
toDrop = []
for i in range(len(MABMainDF)):
    if MABMainDF.loc[i, "tournament"] == "TS" and MABMainDF.loc[i, "numPlayers"] == 32:
        toDrop.append(i)
MABMainDF.drop(toDrop, inplace=True)

RRDF = filterDF(classicalDF, "tournament", ["RR1", "RR5"])

MABDF = MABMainDF
for col in MABDF.columns:
    if col not in RRDF.columns:
        MABDF.drop(columns=[col], inplace=True)
for col in RRDF.columns:
    if col not in MABDF.columns:
        RRDF.drop(columns=[col], inplace=True)
RRvsMABDF = pd.concat([RRDF, MABDF])

allDF = pd.concat([RRvsMABDF, classicalNonRRDF, sortingDF])
gridLines = False
makeAccuracyPlots(allDF, "numPlayers", f"img/report_images/", True, "Number of Players", legLocs=["upper right", "upper left", "upper left", "upper right"])
