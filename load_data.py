import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_diagram

# load config params and paths
from functions.config import *

# Multi-layer
multilayer = False

# Load clinical data
df = pd.read_excel(os.path.join(basepath, "clinical_data.xlsx"))
print(df)

# change target (only HV vs pwMS)
target = 1 - df["control"].values
print("Target shape is {}".format(target))

# generate filenames to load
filenames = ["{}.csv".format(x) for x in df["id"]]
print("Numer of filenames to be loaded: {}".format(len(filenames)))

if multilayer:
    data = np.zeros(shape=(len(filenames), num_nodes*2, num_nodes*2))
else:
    data = np.zeros(shape=(len(filenames), num_nodes, num_nodes, num_layers))

# load data into np structures
for i, filename in enumerate(filenames):
    df = pd.read_csv(os.path.join(basepath_FA, filename), index_col=0)
    if multilayer:
        data[i,76:,:76] = df.values
        data[i,:76,76:] = df.values
    else:
        data[i, :, :, 0] = df.values
    
    df = pd.read_csv(os.path.join(basepath_GM, filename), index_col=0)
    if multilayer:
        data[i,:76,:76] = df.values
    else:
        data[i, :, :, 1] = df.values
    
    df = pd.read_csv(os.path.join(basepath_RS, filename), index_col=0)
    if multilayer:
        data[i,76:,76:] = df.values
    else:
        data[i, :, :, 2] = df.values

print("data shape: {}".format(data.shape))

# export data
out = os.path.join(tmp_basepath, "data.npy")
np.save(out, data, allow_pickle=True)
print("Data exported at {}".format(out))

out = os.path.join(tmp_basepath, "target.npy")
np.save(out, target, allow_pickle=True)
print("Target exported at {}".format(out))