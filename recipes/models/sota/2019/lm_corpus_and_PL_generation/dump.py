import sys

prev_line = "hello world"
for _, line in enumerate(sys.stdin):
    line = line.strip()
    if prev_line != "":
        print(line, end=" ")
        prev_line = line
    else:
        print("\n" + line, end=" ")
        prev_line = line
