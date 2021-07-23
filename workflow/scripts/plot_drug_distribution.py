from utils import plot_node_edge_combined, load_pickle


data = load_pickle(snakemake.input.drugs)
fig = plot_node_edge_combined(data)

fig.write_html(snakemake.output.density_html)
fig.write_image(snakemake.output.density_png)
fig.write_json('slides/drug_distribution.json')

