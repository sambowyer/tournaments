from typing import List, Dict

def writeHeaders(filename : str, headers : List[str]):
    with open(filename, "a") as f:
        for col in headers:
            f.write(col)
            f.write(",")
        f.write("\n")

def writeStatsCollectionToCSV(statsCollection : List[Dict], filename : str, headers=False):
    with open(filename, "a") as f:
        if headers:
            for col in statsCollection[0]:
                f.write(col)
                f.write(",")
            f.write("\n")

        for stats in statsCollection:
            for key in stats:
                f.write(f"{stats[key]},")
        
            f.write("\n")