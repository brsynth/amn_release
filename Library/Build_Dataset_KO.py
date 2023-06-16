###############################################################################
# This library create training sets for AMN
# Trainig sets are either based on experimental datasets
# or FBA (Cobrapy) simulations
# Authors: Jean-loup Faulon jfaulon@gmail.com and Bastien Mollet
###############################################################################

from __future__ import print_function
import os
import sys
import csv
import random
import math
import numpy as np
import pandas
import time
import json
import copy
import pickle
import cobra
# import cobra.test # was crashing the colab implementation
import cobra.manipulation as manip
from cobra import Reaction, Metabolite, Model
from cobra.flux_analysis import pfba
from sklearn.utils import shuffle
sys.setrecursionlimit(10000) # for row_echelon function

###############################################################################
# IOs with pandas
###############################################################################

def read_csv(filename):
    # Reading datafile with pandas
    # Return HEADER and DATA
    filename += '.csv'
    dataframe = pandas.read_csv(filename, header=0)
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:,:])
    return HEADER, DATA

def MinMaxScaler(data, Min_Scaler = -1.0e12, Max_Scaler = 1.0e12):
    # MinMax standardize np array data
    if Max_Scaler == 1.0e12: # Scale
        Min_Scaler, Max_Scaler = np.min(data), np.max(data)
        data = (data-Min_Scaler)/(Max_Scaler-Min_Scaler)
    else: # Descale
        data = data * (Max_Scaler-Min_Scaler) + Min_Scaler
        Min_Scaler = -1.0e12
        Max_Scaler = 1.0e12      
    return data, Min_Scaler, Max_Scaler

def MaxScaler(data, Max_Scaler = 1.0e12):
    # Max standardize np array data
    if Max_Scaler == 1.0e12: # Scale
        Max_Scaler = np.max(data)
        data = data/Max_Scaler
    else: # Descale
        data = data * Max_Scaler
        Max_Scaler = 1.0e12      
    return data, Max_Scaler

def read_XY(filename, nX, nY=1, scaling=''):
    # Format data for training
    # Function read_training_data is defined in module (1)
    # if scaling == 'X' X is scaled
    # if scaling == 'Y' Y is scaled
    # if scaling == 'XY' X and Y are scaled
    _, XY = read_csv(filename)
    XY = np.asarray(XY)
    X = XY[ : ,    : nX]
    Y = XY[ : , nX : nX+nY]
    X, _, _ = MinMaxScaler(X) if scaling == 'X' or scaling == 'XY' else X, 0, 0
    Y, _, _ = MinMaxScaler(Y) if scaling == 'Y' or scaling == 'XY' else Y, 0, 0
    return X,Y

###############################################################################
# Cobra's model utilities and matrices (written by Bastien Mollet)
###############################################################################

# Cobra utilities and stoichiometric derived matrices
def get_index_from_id(name,L):
    # Return index in L of id name
    for i in range(len(L)):
        if L[i].id == name:
            return i
    return -1

def get_objective(model):
    # Get the reaction carring the objective
    # Someone please tell me if there is
    # a clearner way in Cobra to get
    # the objective reaction

    r = str(model.objective.expression)
    r = r.split()
    r = r[0].split('*')
    obj_id = r[1]

    # line below crash if does not exist
    r = model.reactions.get_by_id(obj_id)

    return obj_id


def get_LB_UB(model, Pin, mediumbound, X):
    # Default bounds for the model, will be used as template vector on which
    # we set bounds specific to each training set element
    def_UB = []
    def_LB = []
    for r in model.reactions:
        def_UB.append(r.upper_bound)
        def_LB.append(r.lower_bound)

    # Replace default by X values for Upper Bounds (UB case)
    # or both UB and LB for Exact Bounds (EB case)
    X_copied = X.copy()
    X_copied[X_copied==0] = -1 # -1 for real 0 bounds, others are let to default bounds
    x_bounds = np.matmul(X_copied, Pin)
    UB = []
    LB = []
    for x_bound in x_bounds:
        i=0
        this_UB = def_UB
        this_LB = def_LB
        for bound in x_bound:
            if bound == 0: # if the value in x_bound is 0, we keep the default value
                continue
            if bound < 0: # if the value in x_bound is -1, we force the bounds to 0 (absence of compound)
                this_UB[i] = 0.0
                this_LB[i] = 0.0 # should be this value by default but here to make sure
            if bound > 0: # if the value in x_bound is > 0, we change the default value to the x_bound value
                # in both EB and UB cases we set the upper bound to the x_bound value if >0 
                this_UB[i] = bound
            if mediumbound == 'EB' and bound > 0: # same for the lower bound, only in EB case
                this_LB[i] = bound
            i += 1
        UB.append(this_UB)
        LB.append(this_LB)
    
    return LB, UB


def get_matrices(model, medium, measure, reactions):
    # Get matrices for AMN_QP and AMN_Wt
    # Return
    # - S [mxn]: stochiometric matrix
    # - V2M [mxn]: to compute metabolite
    #        production fluxes from reaction fluxes
    # - M2V [mxn]: to compute reaction fluxes
    #        from substrate production fluxes
    # - Pin [n_in x n]: to go from reactions to medium fluxes
    # - Pout [n_out x n]: to go from reactions to measured fluxes

    # m = metabolite, n = reaction/v/flux, p = medium
    S = np.asarray(cobra.util.array.create_stoichiometric_matrix(model))
    n, m, n_in, n_out = S.shape[1], S.shape[0], len(medium), len(measure)

    # Get V2M and M2V from S
    V2M, M2V = S.copy(), S.copy()
    for i in range(m):
        for j in range(n):
            if S[i][j] < 0:
                V2M[i][j] = 0
                M2V[i][j] = -1/S[i][j]
            else:
                V2M[i][j] = S[i][j]
                M2V[i][j] = 0
    M2V = np.transpose(M2V)

    # Boundary matrices from reaction to medium fluxes
    Pin, i = np.zeros((n_in,n)), 0
    for rid in medium:
        j = get_index_from_id(rid,reactions)
        Pin[i][j] = 1
        i = i+1

    # Experimental measurements matrix from reaction to measured fluxes
    Pout, i = np.zeros((n_out,n)), 0
    for rid in measure:
        j = get_index_from_id(rid,reactions)
        Pout[i][j] = 1
        i = i+1

    return S, Pin, Pout, V2M, M2V

def row_echelon(A,C):
    # Return Row Echelon Form of matrix A and the matrix C 
    # will be used to perform all the operations on b later
    # This function is recursive, it works by turning the first 
    # non-zero row to 1. Then substract all the other row
    # to turn them to 0. Thus, perform the same operation on 
    # the second row/ second column.
    # If matrix A has no columns or rows, it is already in REF, 
    # so we return itself, it's the end of the recursion.

    r, c = A.shape
    if r == 0 or c == 0:
        return A,C

    # We search for non-zero element in the first column.
    # (If/else is used in a strange wy but the Else is skipped 
    # if break happens in if)
    #( Else can't be used in the for)
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # If all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:,1:],C)
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B[0]]),C

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        C_ith_row = C[i].copy()
        A[i] = A[0]
        C[i] = C[0]
        C[0] = C_ith_row
        A[0] = ith_row

    # We divide first row by first element in it
    # Here it's important to first change C as the value
    Scaling_factor = A[0,0] # Keep this value in memory in case it makes too high values.
    C[0] = C[0] / Scaling_factor
    A[0] = A[0] / Scaling_factor

    # We subtract all subsequent rows with first row 
    # (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    C[1:] -= C[0] * A[1:,0:1]
    A[1:] -= A[0] * A[1:,0:1]

    #### Controling values to remain differentiable ####
    up_bound = np.amax(A[1:],1)
    for i in range(1,len(up_bound)):
        max_row = up_bound[i-1]
        if max_row >=1000:
            C[i] =  C[i] / max_row
            A[i] = A[i] / max_row

    # If the scaling factor is too small, values in A[0] can be too high
    if np.amax(A[0]) >= 1000:
        C[0] = C[0] * Scaling_factor
        A[0] = A[0] * Scaling_factor
    #### End of the controling part ####

    # we perform REF on matrix from second row, from second column
    B = row_echelon(A[1:,1:],C[1:,:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B[0]]) ]),\
            np.vstack([C[:1],  B[1]])

def get_B(model, S, medium, verbose=False):
    # A matrix used to get boundary vectors in get_matrices_LP
    n, m, p = S.shape[1], S.shape[0], len(medium)
    B, i = np.zeros((p,m)), 0
    # print(p)
    for rid in medium:
        k = get_index_from_id(rid,model.reactions)
        r = model.reactions[k]
        # print(r.products)
        p = r.products[0] # medium reactions have only one product
        j = get_index_from_id(p.id,model.metabolites)
        B[i][j] = 1
        i = i+1
    if verbose: print("When you get B: ", B[0], B.shape)
    # print("Where is B non-zero: ", np.nonzero(B))
    return B

def get_matrices_LP_empty():
    S_int, S_ext, Q, P, b_int, b_ext, Sb, c = 0, 0, 0, 0, 0, 0, 0, 0
    return S_int, S_ext, Q, P, b_int, b_ext, Sb, c

def get_matrices_LP(model, mediumbound, X, S, Pin, medium, objective, 
                     verbose=False):
    # Get matrices and vectors for LP cells from
    # Y. Yang et al. Mathematics & Computers in Simulation 101, 103–112, (2014)
    # Outputs:
    # - Sint [mxn] [m metabolites, n fluxes]
    #   For EB the stoichiometric matrix S where columns corresponding
    #   to intake fluxes are zeroed out
    #   For UB same as EB + rows corresponding to metabolites entries
    #   are zeroed out
    # - Sext [mxn]
    #   For EB = I [nxn] (m=n)
    #   For UB the stoichiometric matrix S where only rows
    #   corresponding to internal metabolites are kept + I -
    #   stoichiometric matrix S where only rows
    #   corresponding to internal metabolites are kept
    # - Q = S_int^T (S_int S_int^T)-1 [n x m]
    # - P = Q S_int - I [n x n]
    # - b_int [m]
    #   For EB the extact bound values
    #   For UB = 0
    # - b_ext [m]
    #   For EB = 0
    #   For UB the upper bound values
    # columns in Sb corresponding to medium are zeroed out

    Sb = -np.transpose(S.copy())
    S_int = Sb.copy()

    c = np.zeros(S.shape[1])
    for i in range(len(objective)):
        c[get_index_from_id(objective[i],model.reactions)] = -1.0 
        # Here this parameter can be tuned to increase the focus on maximizing c
    c  = np.float32(c)

    inputs = np.float32(X)

    if inputs.shape[1] == S.shape[1]: # Special case inputs = Vsol * noise
        V_all = inputs
    else:  # V =  Pin inputs
        Pin  = np.float32(Pin)
        V_all = np.matmul(inputs, Pin)
        # V_all = V_all.numpy()

    for rid in medium:
        i = get_index_from_id(rid,model.reactions)
        Sb[i] = np.zeros(S.shape[0])

    if mediumbound == 'UB':
        # print('We are in UB')
        S_ext =  Sb.copy()
        for rid in medium:
            i = get_index_from_id(rid,model.reactions)
            r = model.reactions[i]
            # We need to access metabolites because S_int
            # stand for the metabolites that are producted internally
            # Whereas S_ext is the stoechiometric matrix for the 
            # metabolites that have
            # external intake.
            p = r.products[0] # medium reactions have only one product
            j = get_index_from_id(p.id,model.metabolites)
            #S_int[i] = np.zeros(parameter.S.shape[0])
            #S_ext[:,j] = -parameter.S[j]
            # S_int[:,j] = np.zeros(S.shape[1])
        I = -np.identity(S_int.shape[0])
        S_ext_p = -np.copy(S_ext) 
        #This part of S_ext ensure that every flux is positive.
        S_ext = np.concatenate((S_ext, I), axis=1)
        S_ext = np.concatenate((S_ext, S_ext_p), axis=1)
    else:
        # print('We are in EB')
        S_int = Sb.copy()
        S_ext = -np.identity(S_int.shape[0])

    # Triangulate matrix S_int and record row permutation in Transform
    S_int = np.transpose(S_int)
    S_int, Transform = row_echelon(S_int, np.identity(S_int.shape[0])) 
    S_int = S_int[~np.all(S_int == 0, axis=1)] # remove zero line

    # print("transform:", Transform.shape)
    # P and Q
    Q = np.dot(S_int, np.transpose(S_int))
    Q = np.linalg.inv(Q) # inverse matrix
    Q = np.dot(np.transpose(S_int), Q)

    P = np.dot(Q, S_int)
    P = P - np.identity(P.shape[0]) # -(I-P)

    # b_int and b_ext
    B = get_B(model, S, medium, verbose=verbose)
    b = np.matmul(inputs, B)
    b = np.float32(b)

    if mediumbound == 'UB':
        b_int = np.zeros(S_int.shape[0])  # null vector
        # b_int[np.where(b_int==0)] = DEFAULT_UB # breaks the method
        b_int = np.float32(b_int)
        b_ext_all = b
        # This part aims to build the b vector that can be used with 2014. 
        # It takes the same input as 2006 but it needs
        # to be added parts with 0 to ensure the different inequalities.
        # As explained for M, b_ext in the UB case ensure 3 constraints.
        # The first one (upper bounds) is set by b_ext.
        # b_add aims to ensure the next two ones.
        new_b_ext = []
        for i in range(len(V_all)):
            V = V_all[i]
            b_ext = b_ext_all[i]
            if verbose: print("b_ext before b_add ", b_ext.shape)
            b_add = np.zeros(V.shape[0] + b_ext.shape[0])
            if 'ATPM' in model.reactions:
                # ATPM is the only reaction (to our knowledge) 
                # with a lower bound.
                # It could be a good update to search for non-zero 
                # lower bounds automatically.
                indice = get_index_from_id('ATPM', model.reactions)
                ATPM_LB = model.reactions.get_by_id('ATPM').lower_bound
                b_add[indice] = -ATPM_LB
            # print(b_add)
            b_add = np.transpose(b_add)
            b_ext = np.transpose(b_ext)
            b_used = np.concatenate([b_ext,b_add], axis=0)
            if verbose: print("b_ext after b_add ", b_used)
            new_b_ext.append(b_used)
        b_ext = np.array(new_b_ext, dtype=np.float32)

    else: # EB b_int must be transformed because S_int was passed in row form
        b_int = np.matmul(np.float32(Transform),b.T)
        b_int = np.transpose(b_int[:S_int.shape[0]])
        b_ext = np.zeros(S.shape[1])  # null vector
        # b_ext[np.where(b_ext==0)] = DEFAULT_UB # breaks the method
        b_ext = np.float32(b_ext)

    Sb = np.float32(Sb)
    S_int = np.float32(S_int)
    S_ext = np.float32(S_ext)
    Q  = np.float32(Q)
    P  = np.float32(P)
    return S_int, S_ext, Q, P, b_int, b_ext, Sb, c

def reduce_model(model, medium, measure, flux, verbose=False):
    # Remove all reactions not in medium having a zero flux
    # Input: the model, the medium, the flux vector (a 2D array)
    # Output: the reduce model

    # Collect reaction to be removed
    remove = {}
    for i in range(flux.shape[1]):
        if np.count_nonzero(flux[:,i]) == 0 and \
        model.reactions[i].id not in medium and \
        model.reactions[i].id not in measure:
            remove[i] = model.reactions[i]

    # Actual deletion
    model.remove_reactions(list(remove.values()))
    manip.delete.prune_unused_reactions(model)
    for m in model.metabolites:
        if len(m.reactions) == 0:
            model.remove_metabolites(m)
    manip.delete.prune_unused_metabolites(model)
    print('reduced numbers of metabolites and reactions:',
          len(model.metabolites), len(model.reactions))

    return model

###############################################################################
# Running Cobra
###############################################################################

def run_cobra(model, objective, IN, method='FBA', verbose=False,
              objective_fraction=0.75, cobra_min_flux=1.0e-8,
              gene_KO=''):
    # Inputs:
    # - model
    # - objective: a list of reactions (first two only are considered)
    # - IN: Initial values for all reaction fluxes
    # - method: FBA or pFBA
    # run FBA optimization to compute recation fluxes on the provided model
    # set the medium using values in dictionary IN.
    # When 2 objectives are given one first maximize the first objective (obj1).
    # then one set the upper and lower bounds for that objective to
    # objective_fraction * obj1 (e.g. objective_fraction = 0.75) and maximize
    # for the second objective
    # Outputs:
    # - FLUX, the reaction fluxes compyted by FBA for all reactions
    # - The value for the objective

    # set the medium and objective
    
    with model:
        medium = model.medium # This is the model medium
        medini = medium.copy()
        for k in medium.keys(): # Reset the medium
            medium[k] = 0
        for k in IN.keys(): # Additional cmpds added to medium
            if k in medium.keys():
                medium[k] = float(IN[k])
            else:
                # only modify the bound on internal fluxes if KO (=0)
                if IN[k] == 0.0001:
                    model.reactions.get_by_id(k).bounds = (0, 0)
                    # print("KO of reaction ", k, ", bounds are set to : ", model.reactions.get_by_id(k).bounds)
        model.medium = medium

        # Simulate KO if any:
        if gene_KO != '':
            model.genes.get_by_id(gene_KO).knock_out()

        # run FBA for primal objective
        model.objective = objective[0]

        # print("searching solution...")
        try:
            solution = cobra.flux_analysis.pfba(model) \
            if method == 'pFBA' else model.slim_optimize(error_value=0.0)
            # print("found!")
        except cobra.exceptions.Infeasible:
            return [0 for x in range(len(model.reactions))], 0.0

        if method == 'pFBA':
            solution_val = solution.fluxes[objective[0]]
        else:
            solution_val = solution
        if verbose:
            print('primal objectif =', objective, method, solution_val)

        # run FBA for second objective
        # primal objectif is set to a fraction of its value
        if len(objective) > 1:
            obj = model.reactions.get_by_id(objective[0])
            obj_lb, obj_ub = obj.lower_bound, obj.upper_bound
            obj.lower_bound = objective_fraction * solution_val
            obj.upper_bound = objective_fraction * solution_val
            model.objective = objective[1]
            solution = cobra.flux_analysis.pfba(model) \
            if method == 'pFBA' else model.optimize()
            solution_val = solution.fluxes[objective[1]]
            if verbose:
                print('second objectif =', objective, method, solution_val)

            # reset bounds and objective to intial values
            obj.lower_bound, obj.upper_bound = obj_lb, obj_ub
            model.objective = objective[0]

        # get the fluxes for all model reactions
        # FLUX = IN.copy()
        if method == 'pFBA':
            FLUX = {r.id:0 for r in model.reactions}
            for x in model.reactions:
                if x.id in FLUX.keys():
                    FLUX[x.id] = solution.fluxes[x.id]
                    if math.fabs(float(FLUX[x.id])) < cobra_min_flux: # !!!
                        FLUX[x.id] = 0
        else:
            FLUX = [0 for x in range(len(model.reactions))]

        # Reset medium
        model.medium = medini

    return FLUX, solution_val

###############################################################################
# Generating random medium runing Cobra
###############################################################################

def create_random_medium_cobra(model, objective, 
                               medium, mediumbound, in_varmed, levmed, valmed, ratmed,
                               method='FBA', verbose=False,
                               cobra_min_objective=1.0e-3):
    # Generate a random input and get Cobra's output
    # Input:
    # - model
    # - objective: the reaction fluxes to optimize
    # - medium: list of reaction fluxes in medium
    # - in_varmed: the medium reaction fluxes allowed to change
    #              (can be empty then varmed are drawn at random)
    # - levmed: teh number of level a flux can take
    # - valmed: the maximum value the flux can take
    # - ratmed: the ration of fluxes turned on
    # - method: the method used by Cobra
    # Make sure the medium does not kill the objective
    # i.e. objective > cobra_min_objective
    # Ouput:
    # - Intial reaction fluxes set to medium values

    MAX_iteration = 5 # max numbrer of Cobra's failaure allowed

    medini = model.medium.copy()
    INFLUX = {}
    for r in model.reactions:
        INFLUX[r.id] = 0

    # X = actual number of variable medium turned ON
    L_in_varmed = len(in_varmed) 
    if L_in_varmed > 0:
        X = len(in_varmed)
    else:
        X = sum(map(lambda x : x>1, levmed)) # total number of variable medium
        X = np.random.binomial(X, ratmed, 1)[0] if ratmed < 1 else int(ratmed)
        X = 1 if X == 0 else X
    
    # Indices for minmed varmed
    minmed, varmed = [], []
    for i in range(len(medium)):
        if levmed[i] <= 1: # mimimum medium indices
            minmed.append(i)
        else:
            if len(in_varmed) > 0:
                if medium[i] not in in_varmed:
                    continue
            varmed.append(i) # variable medium indices

    modmed = minmed + varmed  if mediumbound == 'EB' else varmed
    
    for iteration in range(MAX_iteration):
        # create random medium choosing X fluxes in varmed at random
        INFLUX = {k: 0 for k in INFLUX.keys()} # reset
        model.medium = medini # reset
        varmed = shuffle(varmed) # that's where random choice occur
        for i in range(len(minmed)):
            j = minmed[i]
            k = medium[j]
            INFLUX[k], model.medium[k] = valmed[j], valmed[j]
        for i in range(X):
            j = varmed[i]
            k = medium[j]
            v = (L_in_varmed+1) * np.random.randint(1,high=levmed[j]) * valmed[j]/(levmed[j]-1)
            INFLUX[k], model.medium[k] = v, v

        # check with cobra
        try:
            _, obj = run_cobra(model, objective, INFLUX,
                               method=method, verbose=False)
        except:
            print('Cobra cannot be run start again')
            treshold, iteration, up, valmed = \
            init_constrained_objective(objective_value, in_treshold, 
                            modmed, valmed, verbose=verbose)
            continue
            
        if obj < cobra_min_objective:
            continue # must have some objective

        # We have a solution
        if verbose:
            p = [ medium[varmed[i]] for i in range(X)]
            print('pass (varmed, obj):', p, obj)
        break

    model.medium = medini # reset medium
    
    return INFLUX

def get_io_cobra(model, objective, 
                 medium, mediumbound, varmed, levmed, valmed, ratmed,
                 E, method='FBA', inf={}, verbose=False):
    # Generate a random input and get Cobra's output
    # Input:
    # - model: the cobra model
    # - objective: the list of objectiev fluxes to maximize
    # - medium: list of reaction fluxes in medium
    # - varmed: the medium reaction fluxes allowed to change
    #            (can be empty then varmed are drawn at random)
    # - levmed: the number of level an uptake flux can take
    # - valmed: the maximum value the flux can take
    # - ratmed: the ration of fluxes turned on
    # - method: the method used by Cobra
    # Output:
    # - X=medium , Y=fluxes for reactions in E

    if inf == {}:
        inf = create_random_medium_cobra(model, objective, 
                                         medium, mediumbound,
                                         varmed, levmed, valmed.copy(), ratmed,
                                         method=method,verbose=verbose)
    out,obj = run_cobra(model,objective,inf,method=method,verbose=verbose)
    Y = np.asarray(list(out.values()))
    X = np.asarray([ inf[medium[i]] for i in range(len(medium)) ])

    return X, Y

###############################################################################
# Creating, saving and loading training set object
# Training set object used in all modules
###############################################################################

class TrainingSet:
    # All element necessary to run AMN
    # cf. save for definition of parameters
    def __init__(self,  cobraname='',
                 mediumname='', mediumbound='EB', mediumsize=-1,
                 objective=[], method='FBA',
                 measure=[], verbose=False,
                 genesize=-1):

        if cobraname == '':
            return # create an empty object
        if not os.path.isfile(cobraname+'.xml'):
            print(cobraname)
            sys.exit('xml cobra file not found')
        if not os.path.isfile(mediumname+'.csv'):
            print(mediumname)
            sys.exit('medium or experimental file not found')
        self.cobraname = cobraname # model cobra file
        self.mediumname = mediumname # medium file
        self.mediumbound = mediumbound # EB or UB
        self.method = method
        self.model = cobra.io.read_sbml_model(cobraname+'.xml')
        self.reduce = False
        self.allmatrices = True
        
        # set medium
        H, M = read_csv(mediumname)
        if 'EXP' in self.method : # Reading X, Y
            if mediumsize < 1:
                sys.exit('must indicate medium size with experimental dataset')
            medium = []
            for i in range(mediumsize):
                medium.append(H[i])
            self.medium = medium
            self.levmed, self.valmed, self.ratmed = [], [], 0
            if genesize < 1:
                self.X_genes = np.array([[0]])
                self.genes = []
                self.X = M[:,0:len(medium)]
                self.Y = M[:,len(medium):]
            else:
                genes = []
                for i in range(genesize):
                    genes.append(H[len(medium)+i])
                self.genes = genes
                self.X = M[:,0:len(medium)]
                self.X_genes = M[:,len(medium):len(medium)+len(genes)]
                self.Y = M[:,len(medium)+len(genes):]
            self.size = self.Y.shape[0]
        else:
            self.medium = H[1:]
            self.levmed = [float(i) for i in M[0,1:]]
            self.valmed = [float(i) for i in M[1,1:]]
            self.ratmed = float(M[2,1])
            self.X, self.Y = np.asarray([]).reshape(0,0), \
            np.asarray([]).reshape(0,0)

        if verbose:
            print('medium:',self.medium)
            print('levmed:',self.levmed)
            print('valmed:',self.valmed)
            print('ratmed:',self.ratmed)

        # set objectve and measured reactions lists
        self.objective = [get_objective(self.model)] \
        if objective == [] else objective
        self.measure = [r.id for r in self.model.reactions] \
        if measure == [] else measure
        if verbose:
            print('objective: ',self.objective)
            print('measurements size: ',len(self.measure))

        # compute matrices and objective vector for AMN
        self.S, self.Pin, self.Pout, self.V2M, self.M2V = \
        get_matrices(self.model, self.medium, self.measure,
                     self.model.reactions)

    def reduce_and_run(self,verbose=False):
        # reduce a model recompute matrices and rerun cobra
        # with the provided training set
        measure = [] if len(self.measure) == len(self.model.reactions) \
        else self.measure
        self.model = reduce_model(self.model, self.medium, measure,
                                  self.Y, verbose=verbose)
        self.measure = [r.id for r in self.model.reactions] \
        if measure == [] else measure

        self.get(sample_size=self.size, reduce=True, verbose=verbose)

    def save(self, filename, no_matrices_LP=False, reduce=False, verbose=False):
        # save cobra model in xml and parameter in npz (compressed npy)
        self.reduce = reduce
        if self.reduce:
            self.reduce_and_run(verbose=verbose)
        # Recompute matrices
        self.S, self.Pin, self.Pout, self.V2M, self.M2V = \
        get_matrices(self.model, self.medium, self.measure,
                         self.model.reactions)
        
        if no_matrices_LP:
            self.S_int, self.S_ext, self.Q, self.P, \
            self.b_int, self.b_ext, self.Sb, self.c = \
            get_matrices_LP_empty()
        else:
            self.S_int, self.S_ext, self.Q, self.P, \
            self.b_int, self.b_ext, self.Sb, self.c = \
            get_matrices_LP(self.model, self.mediumbound, self.X, self.S,
                                self.Pin, self.medium, self.objective)

        self.LB, self.UB = get_LB_UB(self.model, self.Pin, self.mediumbound, self.X)
        
        # save cobra file
        cobra.io.write_sbml_model(self.model, filename+'.xml')
        # save parameters
        np.savez_compressed(filename, 
                            cobraname = filename,
                            reduce = self.reduce,
                            mediumname = self.mediumname,
                            mediumbound = self.mediumbound,
                            objective =self.objective,
                            method = self.method,
                            size = self.size,
                            medium = self.medium,
                            levmed = self.levmed, 
                            valmed = self.valmed, 
                            ratmed = self.ratmed, 
                            measure = self.measure,
                            S = self.S,
                            Pin = self.Pin,
                            Pout = self.Pout,
                            V2M = self.V2M,
                            M2V = self.M2V,
                            X = self.X,
                            Y = self.Y,
                            S_int = self.S_int,
                            S_ext = self.S_ext,
                            Q = self.Q,
                            P = self.P,
                            b_int = self.b_int,
                            b_ext = self.b_ext,
                            Sb = self.Sb,
                            c = self.c,
                            genes = self.genes,
                            X_genes = self.X_genes,
                            LB = self.LB,
                            UB = self.UB)
        
    def load(self, filename):
        # load parameters (npz format)
        if not os.path.isfile(filename+'.npz'):
            print(filename+'.npz')
            sys.exit('file not found')
        loaded = np.load(filename+'.npz')
        self.cobraname = str(loaded['cobraname'])
        self.reduce = str(loaded['reduce'])
        self.reduce = True if self.reduce == 'True' else False
        self.mediumname = str(loaded['mediumname'])
        self.mediumbound = str(loaded['mediumbound'])
        self.objective = loaded['objective']
        self.method = str(loaded['method'])
        self.size = loaded['size']
        self.medium = loaded['medium']
        self.levmed = loaded['levmed']
        self.valmed = loaded['valmed']
        self.ratmed = loaded['ratmed']
        self.measure = loaded['measure']
        self.S = loaded['S']
        self.Pin = loaded['Pin']
        self.Pout = loaded['Pout']
        self.V2M = loaded['V2M']
        self.M2V = loaded['M2V']
        self.X = loaded['X']
        self.Y = loaded['Y']
        self.S_int = loaded['S_int']
        self.S_ext = loaded['S_ext']
        self.Q = loaded['Q']
        self.P = loaded['P']
        self.b_int = loaded['b_int']
        self.b_ext = loaded['b_ext']
        self.Sb = loaded['Sb']
        self.c = loaded['c']
        self.allmatrices = True
        self.model = cobra.io.read_sbml_model(self.cobraname+'.xml')
        self.genes = loaded['genes']
        self.X_genes = loaded['X_genes']
        self.LB = loaded['LB']
        self.UB = loaded['UB']

    def printout(self,filename=''):
        if filename != '':
            sys.stdout = open(filename, 'wb')
        print('model file name:',self.cobraname)
        print('reduced model:',self.reduce)
        print('medium file name:',self.mediumname)
        print('medium bound:',self.mediumbound)
        print('list of reactions in objective:',self.objective)
        print('method:',self.method)
        print('trainingsize:',self.size)
        print('list of medium reactions:',len(self.medium))
        print('list of medium levels:',len(self.levmed))
        print('list of medium values:',len(self.valmed))
        print('ratio of variable medium turned on:',self.ratmed)
        print('list of measured reactions:',len(self.measure))
        print('Stoichiometric matrix',self.S.shape)
        print('Boundary matrix from reactions to medium:',self.Pin.shape)
        print('Measurement matrix from reaction to measures:',self.Pout.shape)
        print('Reaction to metabolite matrix:',self.V2M.shape)
        print('Metabolite to reaction matrix:',self.M2V.shape)
        print('Training set X:',self.X.shape)
        print('Training set Y:',self.Y.shape)
        print('list of genes targeted:',len(self.genes))
        print('Training set X for genes:',self.X_genes.shape)
        if self.allmatrices:
            print('S_int matrix', self.S_int.shape)
            print('S_ext matrix', self.S_ext.shape)
            print('Q matrix', self.Q.shape)
            print('P matrix', self.P.shape)
            print('b_int vector', self.b_int.shape)
            print('b_ext vector', self.b_ext.shape)
            print('Sb matrix', self.Sb.shape)
            print('c vector', self.c.shape)
        if filename != '':
            sys.stdout.close()

    def get(self, sample_size=100, varmed=[], reduce=False, verbose=False):
        # Generate a training set for AMN
        # Input: sample size
        # objective_value and variable medium
        # (optional when experimental datafile)
        # Output: X,Y (medium and reaction flux values)

        X, Y, inf = {}, {}, {}
        for i in range(sample_size):
            if verbose: print('sample:',i)

            # Cobra is run on reduce model where X is already know
            if reduce:
                inf = {r.id: 0 for r in self.model.reactions}
                for j in range(len(self.medium)):
                    inf[self.medium[j]] = self.X[i,j]

            X[i], Y[i] = \
            get_io_cobra(self.model, self.objective,
                         self.medium, self.mediumbound, varmed,
                         self.levmed, self.valmed, self.ratmed,
                         self.Pout, inf=inf, method=self.method,
                         verbose=verbose)
        X = np.asarray(list(X.values()))
        Y = np.asarray(list(Y.values()))

        # In case mediumbound is 'EB' replace X[i] by Y[i] for i in medium
        if self.mediumbound == 'EB':
            i = 0
            for rid in self.medium:
                j = get_index_from_id(rid, self.model.reactions)
                X[:,i] = Y[:,j]
                i += 1

        # In case 'get' is called several times
        if self.X.shape[0] > 0 and reduce == False:
            self.X = np.concatenate((self.X, X), axis=0)
            self.Y = np.concatenate((self.Y, Y), axis=0)
        else:
            self.X, self.Y = X, Y
        self.size = self.X.shape[0]
        
    def filter_measure(self, measure=[], verbose=False):
        # Keep only reaction fluxes in measure
        # Input:
        # - measure: a list of measured reaction fluxes
        # - reduce: when True the matrices are reduced considering
        #   the training set, all reactions not in the medium and
        #   having zero flux for all instances in the trainig set
        #   are removed
        # Output:
        # - updated self.Y (reduced to reaction fluxes in measure)
        # - self.Yall all reactions
        
        self.measure = measure if len(self.measure) > 0 else self.measure
        _, _, self.Pout, _, _ = \
        get_matrices(self.model, self.medium, self.measure, self.model.reactions)
        self.Yall = self.Y.copy()
        if self.measure != []:
            # Y = only the reaction fluxes that are in Vout
            Y = np.matmul(self.Y,np.transpose(self.Pout)) \
            if ('EXP') not in self.method else self.Y
            self.Y = Y
        if verbose:
            print('number of reactions: ', self.S.shape[1], self.Yall.shape[1])
            print('number of metabolites: ', self.S.shape[0])
            print('filtered measurements size: ',self.Y.shape[1])
