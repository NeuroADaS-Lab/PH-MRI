import numpy as np

def get_data(data, multilayer=False, connectivity_type=0):
    xx = []
    for i in range(data.shape[0]):
        if multilayer:
            x = data[i, :, :]
        else:
            x = data[i, :, :, connectivity_type]
        
        np.fill_diagonal(x, 0)
        xx.append(x)

    return xx
