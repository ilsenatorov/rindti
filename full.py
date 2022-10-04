import os


def exract_chain():
    pass


glycans = []
lectin_dict = {}
# omega_lectins = os.lisdir("/scratch/SRATCH_SAS/roman/rindti/datasets/oracle/raw/omegafold")
with open("/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle/raw/Lectins.fasta", "r") as lectins:
    lectin_lines = lectins.readlines()
    for i in range(0, len(lectin_lines), 2):
        lectin_dict[lectin_lines[i + 1].strip()] = lectin_lines[i].strip()[1:]


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

            seq = parts[-1]
            if seq not in lectin_dict:
                continue
            lectin = lectin_dict[seq]
            print(f"{lectin}\t{seq}", file=prot)

#            if lectin + ".pdb" in omega_lectins:
#                shutil.copy(f"/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle/raw/omegafold{lectin}.pdb", f"/scratch/SCRATCH_SAS/roman/rindti/datasets/oracle_esm/structures/{lectin}.pdb")
#            else:
#                extract_chain
#

            counter = 0
            for j, v in enumerate(parts[:-1]):
                if len(v) >= 1:
                    counter += 1
                    print(f"Gly{(j + 1):05d}\t{lectin}\t{v}", file=inter)
            print(f"{lectin}\t{counter}", file=counts)
