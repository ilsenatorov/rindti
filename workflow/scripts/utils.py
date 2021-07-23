import pickle

import plotly.express as px
import plotly.graph_objects as go


def load_pickle(filename, graph=True):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        if graph:
            data['nnodes'] = data['data'].apply(nnodes)
            data['nedges'] = data['data'].apply(nedges)
        return data


def nnodes(data):
    return data['x'].size(0)


def nedges(data):
    return data['edge_index'].size(1)//2


def plot_node_edge_scatter(data):
    return px.scatter(
        data,
        'nnodes',
        'nedges',
        labels={'nnodes': 'Number of nodes', 'nedges': 'Number of edges'},
    )


def plot_node_edge_density(data):
    fig = px.density_contour(data, 'nnodes', 'nedges', labels={'nnodes': 'Number of nodes',
                                                               'nedges': 'Number of edges'})
    fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    return fig


def plot_node_edge_combined(data):
    data['text'] = data.index.to_series() + '; Nodes: ' + data.nnodes.astype(str) + '; Edges: ' + data.nedges.astype(str)

    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=data.nnodes,
        y=data.nedges,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=data.nnodes,
        y=data.nedges,
        mode='markers',
        text=data.text,
        hoverinfo='text',
        marker=dict(
            color='rgba(255,255,255,0.7)',
            size=4
        )
    ))
    fig.update_xaxes(title_text='Number of nodes')
    fig.update_yaxes(title_text='Number of edges')
    return fig


def plot_sankey(dims):
    bad_color = 'rgba(230, 50, 50, 0.5)'
    good_color = 'rgba(30, 100, 200, 0.5)'
    curval = dims['GLASS dataset']

    nodes = {
        'label': ['Not Good'],
        'pad': 15,
        'thickness': 15,
        'line': dict(color="black", width=0.5),
    }
    links = {
        'source': [],
        'target': [],
        'value': [],
        'color': [],
        'label': [],
    }
    for i, (key, val) in enumerate(dims.items(), start=1):
        nodes['label'].append(key)
        links['source'].append(i)
        links['target'].append(i+1)
        links['target'].append(0)
        links['value'].append(val)
        links['value'].append(curval-val)
        links['color'].append(good_color)
        links['color'].append(bad_color)
        links['label'].append(key)
        links['label'].append('Not ' + key)
        curval = val

    nodes['label'].append('Good')
    return go.Figure(
        data=[go.Sankey(node=nodes, link=links, textfont=dict(size=12))]
    )
