The distribution of number of nodes and number of edges for the protein contact graphs in the dataset.

Each node is a residue; an edge joins two residues whose C-alpha atoms lie within the contact
cutoff. The cutoff was {{ snakemake.config['prots']['features']['distance']['threshold'] }} Angstrom,
over structures prepared with the ``{{ snakemake.config['prots']['structs']['method'] }}`` method.
