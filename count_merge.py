folder = "method"
names = ["delete", "obfuscate"]
aa_counter = {}

for name in names:
    with open(f"{folder}/counts/aa_count_{name}.tsv", "r") as data:
        for line in data.readlines():
            aa, count, act = line.strip().split("\t")
            if aa not in aa_counter:
                aa_counter[aa] = [0, 0]
            aa_counter[aa][0] += int(count)
            aa_counter[aa][1] += float(act)

with open(f"{folder}/counts/aa_count.tsv", "w") as out:
    for k in aa_counter.keys():
        print(f"{k}\t{aa_counter[k][0]}\t{aa_counter[k][1]}", file=out)
        
aa_counter = {}

for name in names:
    with open(f"{folder}/distance/aa_dist_{name}.tsv", "r") as data:
        for line in data.readlines():
            aa, count, act = line.strip().split("\t")
            aa = int(float(aa))
            if aa not in aa_counter:
                aa_counter[aa] = [0, 0]
            aa_counter[aa][0] += int(count)
            aa_counter[aa][1] += float(act)

max_key = max(aa_counter.keys())
for i in range(max_key):
    if i not in aa_counter:
        aa_counter[i] = [0, 0]

with open(f"{folder}/distance/aa_dist.tsv", "w") as out:
    for k in aa_counter.keys():
        print(f"{k}\t{aa_counter[k][0]}\t{aa_counter[k][1]}", file=out)
