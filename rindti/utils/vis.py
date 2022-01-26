import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_loss_count_dist(losses: dict) -> Figure:
    """Plot distribution of times sampled vs avg loss of families"""
    fig = plt.figure()
    plt.xlabel("Times sampled")
    plt.ylabel("Avg loss")
    plt.title("Prot statistics")
    count = [len(x) for x in losses.values()]
    mean = [np.mean(x) for x in losses.values()]
    plt.scatter(x=count, y=mean)
    return fig
