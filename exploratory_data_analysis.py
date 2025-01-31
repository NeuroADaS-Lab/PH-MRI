import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_diagram

# load config params and paths
from functions.config import *

# Load data
data = np.load(os.path.join(tmp_basepath, "data.npy"), allow_pickle=True)
target = np.load(os.path.join(tmp_basepath, "target.npy"), allow_pickle=True)

# Histograma FA
def hist_single_layer(type=None, data=None, inverse=False):
    if type=="FA":
        tmp = data[:, :, :, 0].reshape(-1)
    elif type=="GM":
        tmp = data[:, :, :, 1].reshape(-1)
    elif type=="RS":
        tmp = data[:, :, :, 2].reshape(-1)
    
    # remove values == 0
    tmp = tmp[tmp > 0.01]

    g = sns.displot(tmp, stat="probability", kde=False, bins=50)
    g.set_axis_labels("Edge value", "Probability")
    filename = "hist_" + type + "_inverse" if inverse else "" + ".png"
    plt.savefig(os.path.join(output_basepath, filename), dpi=300, bbox_inches="tight")
    plt.show()

# plot original histograms
hist_single_layer(type="FA", data=data)
hist_single_layer(type="GM", data=data)
hist_single_layer(type="RS", data=data)

# compute inverse of the data
data = 1 - data

# plot inverse histograms
hist_single_layer(type="FA", data=data, inverse=True)
hist_single_layer(type="GM", data=data, inverse=True)
hist_single_layer(type="RS", data=data, inverse=True)
