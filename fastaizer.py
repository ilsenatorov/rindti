import sys

with open(sys.argv[1], "r") as data, open(sys.argv[2], "w") as fasta:
    for line in data.readlines()[1:]:
        idx, seq = line.strip().split("\t")[:2]
        print(f">{idx}", file=fasta)
        print(seq, file=fasta)
