import pandas as pd

from prepare_proteins import aa_encoding, parse_sif


def sort_edge(node1, node2):
    return tuple(sorted([node1, node2]))


def prepare_protein(sif, index):
    nodes, edges = parse_sif(sif)
    res = 't # {index}\n'.format(index=index)
    for idx, row in nodes.iterrows():
        res += "v {idx} {enc}\n".format(idx=idx, enc=aa_encoding[row['resaa'].lower()])
    edges = edges[['node1', 'node2']].drop_duplicates()
    edges['s_edg'] = edges.apply(lambda x: sort_edge(x['node1'], x['node2']), axis=1)
    edges = edges[['s_edg']].drop_duplicates()
    for idx, row in edges.iterrows():
        res += 'e {first} {second} 0 1000\n'.format(first=row['s_edg'][0], second=row['s_edg'][1])
    return res


proteins = pd.Series(list(snakemake.input.rins), name='sif')
proteins = pd.DataFrame(proteins)

res = ''.join(
    prepare_protein(row['sif'], idx) for idx, row in proteins.iterrows()
)

with open(snakemake.output.out, 'w') as file:
    file.write(res)
