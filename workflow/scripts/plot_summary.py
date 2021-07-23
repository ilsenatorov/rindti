import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prepare_proteins import summary


def plot_summary(filename, output, protein):
    '''
    Creates a summary plot for the given sif file of a protein
    '''
    info = summary(filename)

    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.3, 0.4, 0.4],
        vertical_spacing=0.13,
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar", "colspan": 2}, None],
               [{"type": "bar", "colspan": 2}, None]],
        subplot_titles=['Type 1 count', 'Type 2 count', 'Combined type count', 'Amino acid count'])
    fig.add_trace(
        go.Bar(y=info['type1_count'].values, x=info['type1_count'].index),
        row=1, col=1)
    fig.add_trace(
        go.Bar(y=info['type2_count'].values, x=info['type2_count'].index),
        row=1, col=2)

    fig.add_trace(
        go.Bar(y=info['type_count'].values, x=info['type_count'].index),
        row=2, col=1)

    fig.add_trace(
        go.Bar(y=info['aa_count'].values, x=info['aa_count'].index),
        row=3, col=1)

    fig.update_layout(
        width=700,
        height=700,
        margin=dict(r=10, t=100, b=10, l=10),
        showlegend=False,
        title={
            'text': "RIN summary for {protein}".format(protein=protein),
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )
    fig.write_image(output)


if __name__ == '__main__':
    plot_summary(snakemake.input.sif, snakemake.output.png,
                 snakemake.wildcards.protein)
