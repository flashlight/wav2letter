import sys


for _, line in enumerate(sys.stdin):
    line = line.strip()
    if line == "<P>":
        continue
    else:
        print(line)
