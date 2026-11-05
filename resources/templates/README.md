# Kinase template library

`prots.structs.method: bsite` and `template` crop each protein graph to the region near a
ligand-bound template, so they need a library of template structures. The pipeline reads
it from `<source>/templates/`, which the downloaded datasets do not provide — these three
are the ones the published `template` and `bsite` rows were built with.

Each file is one kinase chain plus its single ligand residue, so
`parse_structs.organic_atoms` finds exactly one ligand to measure distances from.

| File | Complex | PDB | Source |
|---|---|---|---|
| `1iep.pdb` | ABL1 – imatinib | [1IEP](https://www.rcsb.org/structure/1IEP) | Nagar *et al.* 2002 |
| `1aq1.pdb` | CDK2 – staurosporine | [1AQ1](https://www.rcsb.org/structure/1AQ1) | Lawrie *et al.* 1997 |
| `1atp.pdb` | PKA – ATP | [1ATP](https://www.rcsb.org/structure/1ATP) | Zheng *et al.* 1993 |

`best_template` picks one per target by TM-score. Over the 379 Davis targets the split was
240 / 75 / 47 for 1ATP / 1IEP / 1AQ1.

## Installing them

```sh
mkdir -p datasets/davis/resources/templates
cp resources/templates/*.pdb datasets/davis/resources/templates/
```

Same for `datasets/kiba/resources/templates/`. Kinase panels only: BindingDB_Kd spans many
protein families, so aligning every target to a kinase template is meaningless there, and
the paper reports no `template`/`bsite` row for it.

## Regenerating them

`strip_templates.py` rebuilds these three from the full RCSB entries, keeping one chain and
one ligand each and cutting 1ATP to the kinase core (residues 40–300) so all three cover
only the conserved kinase domain:

```sh
mkdir -p raw && cd raw
for id in 1IEP 1AQ1 1ATP; do curl -fsSL "https://files.rcsb.org/download/$id.pdb" -o "$id.pdb"; done
cd .. && python resources/templates/strip_templates.py raw resources/templates
```

It asserts that each stripped file contains exactly the one expected ligand, so a changed
RCSB entry fails loudly rather than silently producing a template with no ligand to
measure from. Needs `biotite` (the `workflow` extra).
