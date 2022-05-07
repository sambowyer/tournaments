fixedLines = []
with open("csvs/MABTuningTests.csv", "r") as f:
    for l in f:
        line = l
        if "(" in line:
            while "(" in line:
                start = line.index("(")
                comma = start + line[start:].index(",")
                end   = line.index(")")
                line  = line[:start] + line[start+1:comma] + line[end+1:]

        fixedLines.append(line)

        # if len(fixedLines) == 500:
        #     with open("csvs/MABTuningTestsFIXED.csv", "a") as g:
        #         for line in fixedLines:
        #             g.write(line)
        #             # g.write("\n")
        #     fixedLines = []

with open("csvs/MABTuningTestsFIXED.csv", "a") as g:
    for line in fixedLines:
        g.write(line)
        # g.write("\n")