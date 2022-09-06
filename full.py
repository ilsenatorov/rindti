glycans = []
with open("/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle_esm/resources/tables/lig.tsv", "w") as lig, \
        open("/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle_esm/resources/tables/prot.tsv", "w") as prot, \
        open("/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle_esm/resources/tables/inter.tsv", "w") as inter, \
        open("/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle_esm/resources/tables/counts.tsv", "w") as counts:
    print("Drug_ID", "Target_ID", "Y", sep="\t", file=inter)
    print("Drug_ID", "Drug", "IUPAC", sep="\t", file=lig)
    print("Target_ID", "Target", sep="\t", file=prot)
    print("Target_ID", "Count", sep="\t", file=counts)

    with open("/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle/raw/all_arrays.csv", "r") as data:
        for i, line in enumerate(data.readlines()):
            print(f"\r{i}", end="")
            parts = line.strip().split(",")

            if i == 0:
                for j, p in enumerate(parts):
                    print(f"Gly{(j + 1):05d}\t\t{p}", file=lig)
                glycans = parts
                continue

            lectin = parts[-1]
            print(f"Lec{i:05d}\t{lectin}", file=prot)
            counter = 0
            for j, v in enumerate(parts[:-1]):
                if len(v) >= 1:
                    counter += 1
                    print(f"Gly{(j + 1):05d}\tLec{i:05d}\t{v}", file=inter)
            print(f"Lec{i:05d}\t{counter}", file=counts)
