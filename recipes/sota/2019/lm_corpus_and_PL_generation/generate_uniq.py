import sys


pl_data = []
with open(sys.argv[1], "r") as f:
    for line in f:
        pl_data.append(line.strip())
pl_data = set(pl_data)

with open(sys.argv[2] + ".unique", "w") as f:
    for elem in pl_data:
        f.write(elem + "\n")
