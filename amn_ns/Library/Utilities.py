###############################################################################
# This file provides data scaling utilities
# Author: Jean-loup Faulon jfaulon@gmail.com
# Updates: 24/11/2023
###############################################################################

from Library.Import import *

def MaxScaler(data, Max_Scaler = 1.0e12):
    # Max standardize np array data
    if Max_Scaler == 1.0e12: # Scale
        Max_Scaler = np.max(data)
        data = data/Max_Scaler
    else: # Descale
        data = data * Max_Scaler
        Max_Scaler = 1.0e12
    return data, Max_Scaler
