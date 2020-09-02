import re
import sys


time_degree = {
    "min.": "minutes",
    "deg.": "degrees",
    "sec.": "seconds",
    "hrs.": "hours",
    "hr.": "hour",
}
abbr_mapping = {
    "mr.": "mister",
    "mr": "mister",
    "dr.": "doctor",
    "dr": "doctor",
    "ms.": "miss",
    "ms": "miss",
    "mrs.": "missus",
    "mrs": "missus",
    "vs.": "versus",
    "vs": "versus",
    "st.": "saint",
    "st": "saint",
}
numbers = set("0123456789")
time_set1 = set(":0123456789")
time_set2 = set("/0123456789")

for _, line in enumerate(sys.stdin):
    line = line.strip()
    line = re.sub(" +", " ", line).strip()
    new_line = []
    prev_word = ""
    for word in line.split():
        if (
            word.lower() in time_degree
            and len(set.intersection(numbers, set(prev_word))) > 0
        ):
            new_line.append(time_degree[word.lower()])
        elif len(set(word) - time_set1) == 0:
            for part in word.split(":"):
                new_line.append(part)
        elif len(set(word) - time_set2) == 0:
            for part in word.split("/"):
                new_line.append(part)
        elif word.lower() in abbr_mapping:
            new_line.append(abbr_mapping[word.lower()])
        elif "&c" in word:
            new_line.append(word.replace("&c", " et cetera "))
        else:
            new_line.append(word)
        prev_word = word
    print(" ".join(new_line))
