###############################################################################
# This library provide utilities for building experimental datasets. 
# The library makes use of scipy, pandas, sklearn, pyDOE2, math, 
# For visualization, it makes use of seaborn and matplotlib
# Two main tasks can be performed (examples in Build_Experimental.ipynb):
# 1) Make combinations of variable elements to be tested in experiments
# 2) From raw plate reader data, make a complete training set of growth rates
# Author: Leon Faure, leon.faure@inrae.fr
###############################################################################

# Libraries

import seaborn as sns
import matplotlib.pyplot as plt
from pyDOE2 import *
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
import math

# Next are some constants that are used
# for the whole experimental data processing

# WIN_SIZE: size of the window used for the linear regression
# (growth rate determination)

# plate_indices: the wells names that will be used, in ordered list

# replicates_dic: how the replicates are spread on the plates. 
# Keys are replicate indice, values are lists of wells name

# ref_*comps_file: paths to files containing reference compositions,
# needed to link plate reader data to actual compositions

# dict_*: dictionary with two keys/values, one indicating which 
# reference composition file and the other indicating the indices
# in this file, from which to extract the compositions experimentally tested
# in this plate

# dict_all: compiling all previous 'dict_' dictionaries for easier access 
# with a single key

# outliers_*: lists for each plate run, compiles all outliers found by looking
# at growth curves or directly finding aberrant growth rate values

# outliers_dic: compiling all previous 'outliers_*' lists for easier access
# with a single key

WIN_SIZE = 6

plate_indices = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
 'A9', 'A10', 'A11', 'A12', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'C1', 'C2', 'C3', 'C4',
 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'D1', 'D2',
 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11','D12',
 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10',
 'E11', 'E12', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
 'F9', 'F10', 'F11', 'F12', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6',
 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'H1', 'H2', 'H3', 'H4',
 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12']

replicates_dic = {i:[plate_indices[i], plate_indices[i+12], 
                plate_indices[i+24], plate_indices[i+36],
                plate_indices[i+48], plate_indices[i+60], 
                plate_indices[i+72], plate_indices[i+84]] for i in range(1,11)}

ref_4comps_file = "Dataset_experimental/compositions_4comps.csv"
ref_3comps_file = "Dataset_experimental/compositions_3comps.csv"
ref_2comps_file = "Dataset_experimental/compositions_2comps.csv"
ref_1comps_file = "Dataset_experimental/compositions_1comps.csv"

dict_20220504 = {"DoE_file":ref_1comps_file, "indices":(0,10)}
dict_20220429 = {"DoE_file":ref_2comps_file, "indices":(0,10)}
dict_20220506 = {"DoE_file":ref_3comps_file, "indices":(0,10)}
dict_20220507 = {"DoE_file":ref_4comps_file, "indices":(0,10)}
dict_20220513 = {"DoE_file":ref_2comps_file, "indices":(10,20)}
dict_20220514 = {"DoE_file":ref_3comps_file, "indices":(10,20)}
dict_20220512 = {"DoE_file":ref_4comps_file, "indices":(10,20)}
dict_20220823 = {"DoE_file":ref_3comps_file, "indices":(20,30)}
dict_20220824 = {"DoE_file":ref_4comps_file, "indices":(20,30)}
dict_20220825 = {"DoE_file":ref_3comps_file, "indices":(30,40)}
dict_20220826 = {"DoE_file":ref_4comps_file, "indices":(30,40)}

dict_all = {"20220429":dict_20220429, "20220504":dict_20220504, 
            "20220506":dict_20220506, "20220507":dict_20220507, 
            "20220512":dict_20220512, "20220513":dict_20220513, 
            "20220514":dict_20220514, "20220823":dict_20220823, 
            "20220824":dict_20220824, "20220825":dict_20220825, 
            "20220826":dict_20220826}

outliers_20220429 = ["A2", "D2", "G2", "F2", "A3", "B3", "C3", "D3", "E3",
                    "D4", "E4", "F4", "H4", "A5", "B5", "F5", "G5", "H5", 
                    "A6", "D6", "C6", "H6", "B7", "G7", "F7", "H7", "A8",
                    "F8", "H8", "G8", "A9", "B9", "E9", "F9", "G9", "H9",
                    "A10", "G10", "H10", "F10", "B10", "F11", "H11", "A11"]
outliers_20220504 = ["E2", "F2", "G2", "H2", "E3", "F3", "G3", "H3", "A4",
                    "B4", "E4", "F4", "H4", "B5", "D5", "E5", "F5", "G5",
                    "H5", "G7", "H8", "D8", "A9", "H9", "D10", "E10", "F10",
                    "G10", "B11", "C11", "G11", "H11"]
outliers_20220506 = ["A2", "E2", "G2", "H2", "C3", "D3", "E3", "A4", "B4",
                    "C4", "F4", "A5", "B5", "C5", "G5", "H5", "C6", "G6",
                    "A7", "D7", "E7", "F7", "G7", "H7", "E8", "F8", "H8",
                    "A9", "E9", "F9", "G9"]                 
outliers_20220507 = ["B3", "C3", "D3", "E3", "H3", "G4", "H4", "A5", "B5",
                    "H5", "E5", "A6", "B6", "F6", "H6", "H7", "D8", "G8",
                    "B8","A9", "B9", "C9", "G9", "H9", "B10", "C10", "D10",
                    "E10", "A11"]
outliers_20220512 = ["A2", "B2", "F2", "D3", "H3", "F3", "G3", "C4", "G4",
                    "H4", "B5", "G5", "H5", "D6", "H6", "F7", "G7", "H7", 
                    "H8", "A9", "H9", "B11", "C11", "D11", "E11", "F11", "G11"]
outliers_20220513 = ["G2", "A3", "E3", "G3", "H3", "A4", "F4", "H4", "A5", "G5",
                    "F5", "H5", "E5", "B6", "H6", "A8", "B8", "H8", "A7", "A9",
                    "B9", "F9", "A10", "F10", "A11", "H11"]
outliers_20220514 = ["A2", "H2", "F3", "C4", "C5" "F7", "G7", "H7", "A8", "B8",
                    "C8", "D8", "E8", "C9", "H9", "F10", "G10"]
outliers_20220823 = ["C2", "A3", "D3", "F3", "G3", "H3", "A4", "D4", "E4", "F4",
                    "G4", "H4", "A5", "C5", "D6", "A7", "H7", "G7", "A8", "H8",
                    "C9", "G10", "H10", "A11", "B11", "E11", "H11"]
outliers_20220824 = ["B2", "C2", "D2", "H2", "A3", "H3", "D3", "E3", "A4", "B4",
                    "C4", "F4", "G4", "F5", "C7", "H7", "B8", "C8", "D8", "F8",
                    "D9", "C10", "B10", "A10", "F10", "C11", "E11", "F11", "G11",
                    "H11"]
outliers_20220825 = ["B2", "C2", "D2", "H2", "A3", "D3", "E3", "F3", "G3", "H3",
                    "C4", "D4", "A6", "C6", "D6", "F6", "H6", "B7", "C7", "D7",
                    "E7", "F7", "A8", "B8", "C8", "D8", "E8", "H8", "A10", "E10",
                    "F10", "G10"]
outliers_20220826 = ["G2", "H2", "C3", "D3", "E3", "F3", "H3", "A4", "B4", "E4",
                    "F4", "B5", "E5", "H5", "E6", "G6", "H6", "C7", "B8", "C8",
                    "F8", "G8", "H8", "B9", "C9", "D9", "F9",  "B10", "B11", "D11"]

outliers_dic = {"20220429":outliers_20220429, "20220504":outliers_20220504, 
                "20220506":outliers_20220506, "20220507":outliers_20220507,
                "20220512":outliers_20220512, "20220513":outliers_20220513,
                "20220514":outliers_20220514, "20220823":outliers_20220823, 
                "20220824":outliers_20220824, "20220825":outliers_20220825,
                "20220826":outliers_20220826}

# Functions

def all_combinations(seed, nbr_variables_on, varmed, nbr_levels=2):
    # seed: controls the random state for shuffling the full factorial design
    # nbr_variables_on: how many variables are ON for this full factorial design
    # nbr_levels: how many levels can a variable take
    # In our case nbr_levels=2 by default because we want ON/OFF binary values

    # Returns: A shuffled full factorial design

    ffd = []

    for el in fullfact([2 for el in varmed]):
        if np.count_nonzero(el) == nbr_variables_on:
            ffd.append(list(el.astype(int)))
        
    ffd = shuffle(ffd, random_state=seed)

    return ffd


def curve_smoothing(y):
    # curve smoothing by running average omitting min and max of the window
    # input is y, the time series of the OD measures
    # returns y_smoothed, the same time series after smoothing
    y_smoothed = y[:2]
    
    for i in range(2,len(y)-2):
        
        sub_y = np.array(y[i-2:i+2])
        i_max = np.argmax(sub_y)
        i_min = np.argmin(sub_y)
        sub_y[i_max] = np.nan
        sub_y[i_min] = np.nan
        y_smoothed.append(np.nanmean(sub_y))
        
    y_smoothed += y[-2:]
    return y_smoothed


def growth_rate_determination(data, out, start_stop):
    # data: daraframe containing all raw data
    # out: outliers list, concerned wells will be ignored (pd.NA values)
    # start_stop: dataframe giving the starting and stopping points for the
    # maximum growth rate search
    # returns all the growth rates of the plate run, as a list of lists

    all_grs = []

    for rep in replicates_dic:
    
        grs = []
    
        for col in replicates_dic[rep]:
            if col not in out:
                win_size = WIN_SIZE

                xdata = data["TIME"].to_list()
                ydata = data[col].to_list()

                # ydata = list(np.array(ydata) - BLANK)

                ydata = curve_smoothing(ydata)
                ydata = curve_smoothing(ydata)
                ydata = curve_smoothing(ydata)
                ydata = curve_smoothing(ydata)
                
                start = start_stop[col][0]
                if pd.isna(start):
                    start = 0
                stop = start_stop[col][1]
                if pd.isna(stop):
                    stop = data["TIME"].iloc[-1]

                start_ind = int(start*6)
                stop_ind = int(stop*6)

                xdata = xdata[start_ind:stop_ind]
                ydata = ydata[start_ind:stop_ind]
                
                all_polyfit_grs = [np.polyfit(xdata[ind:ind+win_size], np.log(ydata[ind:ind+win_size]), 1)[0] for ind in range(0,len(xdata)-win_size)]
                
                grs.append(max(all_polyfit_grs)) 
            else:
                
                grs.append(pd.NA)
        all_grs.append(grs)

    return all_grs