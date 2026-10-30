###############################################################################
# This library loads training sets (npz) for AMN-QP and AMN-NS
# and builds the stoichiometric and boundary matrices from a Cobra model
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Updates: - Nov 2023 to compute fix and variable medium with EXP datasets
#          - June 2024 to add kos with EXP datasets
###############################################################################

from Library.Import import *
import cobra

###############################################################################
# Cobra's model utilities and matrices
###############################################################################

def get_index_from_id(idname,L):
    # Return index in L of id name
    for i in range(len(L)):
        if L[i].id == idname:
            return i
    return -1

def get_matrices(model, medium, ko, measure, reactions):
    # Get matrices for AMN_QP
    # Return
    # - S [mxn]: stochiometric matrix
    # - Pin [n_in x n]: to go from reactions to medium fluxes
    # - Pko [n_ko x n]: to go from reactions to ko gene fluxes
    # - Pout [n_out x n]: to go from reactions to measured fluxes

    # m = metabolite, n = reaction/v/flux, p = medium
    S = np.asarray(cobra.util.array.create_stoichiometric_matrix(model))
    n, m = S.shape[1], S.shape[0]
    n_in, n_ko, n_out = len(medium), len(ko), len(measure)

    # Boundary matrices from reaction to medium fluxes
    Pin, i = np.zeros((n_in,n)), 0
    for rid in medium:
        j = get_index_from_id(rid,reactions)
        Pin[i][j] = 1
        i = i+1

    # Boundary matrices from reaction to ko fluxes
    Pko, i = np.zeros((n_ko,n)), 0
    for rid in ko:
        j = get_index_from_id(rid,reactions)
        Pko[i][j] = 1
        i = i+1

    # Experimental measurements matrix from reaction to measured fluxes
    Pout, i = np.zeros((n_out,n)), 0
    for rid in measure:
        j = get_index_from_id(rid,reactions)
        Pout[i][j] = 1
        i = i+1

    return S, Pin, Pko, Pout

def get_minmed_varmed_ko(medium, verbose=False):
    minmed, varmed, ko = {}, {}, {}
    for m in medium.keys():
        (l, v) = medium[m]
        if l == 0:
            ko[m] = (l, v)
        elif l == 1:
            minmed[m] = (l, v)
        elif l > 1:
            varmed[m] = (l, v)
    return minmed, varmed, ko

###############################################################################
# Training set object used in all modules
###############################################################################

class TrainingSet:
    # All elements necessary to run AMN, loaded from an npz file

    def load(self, filename):
        # load parameters (npz format)
        if not os.path.isfile(filename+'.npz'):
            print(filename+'.npz')
            sys.exit('file not found')
        loaded = np.load(filename+'.npz', allow_pickle=True)
        self.cobraname = str(loaded['cobraname'])
        self.reduce = loaded['reduce']
        self.mediumname = str(loaded['mediumname'])
        nammed = loaded['nammed']
        levmed = loaded['levmed']
        valmed = loaded['valmed']
        self.medium = {}
        for i in range(len(nammed)):
            m = nammed[i]
            self.medium[m] = (levmed[i], valmed[i])
        self.mediumexp = loaded['mediumexp']
        self.objective = loaded['objective']
        self.method = str(loaded['method'])
        self.size = loaded['size']
        self.measure = loaded['measure']
        self.S = loaded['S']
        self.Pin = loaded['Pin']
        if 'Pko' in loaded.keys():
            self.Pko = loaded['Pko']
        else:
            self.Pko = np.asarray([])
        self.Pout = loaded['Pout']
        self.X = loaded['X']
        self.Y = loaded['Y']
        self.model = cobra.io.read_sbml_model(self.cobraname+'.xml')

    def filter_measure(self, measure=[], verbose=False):
        # Keep only reaction fluxes in measure
        # Input:
        # - measure: a list of measured reaction fluxes
        # Output:
        # - self.Yall all reactions

        self.measure = measure if len(self.measure) > 0 else self.measure
        minmed, varmed, ko = get_minmed_varmed_ko(self.medium)
        _, _, _, self.Pout = \
        get_matrices(self.model, varmed, ko, self.measure, self.model.reactions)
        self.Yall = self.Y.copy()
        if len(self.measure) > 0:
            # Y = only the reaction fluxes that are in Vout
            print(self.Y.shape, self.Pout.shape)
            Y = np.matmul(self.Y,np.transpose(self.Pout)) \
            if ('EXP') not in self.method else self.Y
            self.Y = Y
        if verbose:
            print(f'number of reactions: {self.S.shape[1]}={self.Yall.shape[1]}')
            print(f'number of metabolites: {self.S.shape[0]}')
            print(f'filtered measurements size: {self.Y.shape[1]}')
