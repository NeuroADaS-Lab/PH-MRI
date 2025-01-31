import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# load config params and paths
from functions.config import *
from functions.functions import *
from functions.PH_MRI import *

# Load data
data = np.load(os.path.join(tmp_basepath, "data.npy"), allow_pickle=True)
target = np.load(os.path.join(tmp_basepath, "target.npy"), allow_pickle=True)

# compute inverse of the data
data = 1 - data

# Multi-layer
multilayer = False
# FA = 0, GM = 1, RS = 2
connectivity_type = 2

connectivity_names = ["FA", "GM", "RS"]
if multilayer:
    connectivity_name = "Multilayer"    
else:
    connectivity_name = connectivity_names[connectivity_type]

# get data
data_tmp = get_data(data, multilayer=False, connectivity_type=2)

# PH and Betti curves
ph = PH_MRI(data_tmp, target)
ph.compute_PH()
ph.compute_Betti_curves()

# plot values for one single subject
subject = 23
ph.plot_Betti_curves_one_subject(23)

# plot all values for each dimension
dimensions = [0,1,2]

for dimension in dimensions:
    ph.plot_Betti_curves_all_subjects(dimension)

# plot mean and std
for dimension in dimensions:
    ph.plot_Betti_curves_mean(dimension, connectivity_name)