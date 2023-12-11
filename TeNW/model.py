
###########################################
################# Imports #################
###########################################

from .mps_network import MPS
from .mpo_network import MPO_template, MPO_TE_ClockModel_NN, MPO_TE_ClockModel_NNN



###########################################
############# Object template #############
###########################################

class model_template:
    def __init__(self, backend: object, L:int, d: int, p_1: float, p_2: float, p_3: float):
        
        # Must have, leave it as it is
        self.BE = backend
        
        # Must have, leave it as it is
        self.d = d
        self.L = L
        
        # Must have: properties list where the model parameters are stored. If no parameters leave empty "[]"
        self.properties = [p_1, p_2, p_3]

        # Must have, change the MPO_template to your time evolution MPO class from "mpo_network.py"
        self.TE_MPO = MPO_template(backend=backend, L=L, d=d, properties=self.properties)

        # Must have, leave it as it is
        self.MPS = MPS(backend=backend, L=L, d=d)

    
    # The following function is a must have:
    def init_MPS(self, optional_condition):

        # Initiate B_list and schmidt value list
        B_list = None # legs: vL, p, vR
        sv_list = None # legs: vL, vR

        # Must have (write to "self.MPS" variables)
        self.MPS.B_list = B_list
        self.MPS.sv_list = sv_list
        return B_list, sv_list
     


##########################################################
############# Clock Model (nearest-Neighbor) #############
##########################################################

class ClockModel_NN: 
    def __init__(self, backend: object, L:int, d: int, g: float, J: float):
        """
        Clock Model for nearest-neighbors with the Hamiltonian

            H = -J \sum_n ( Z_n  Z_{n+1}^\dagger+ \mathrm{h.c.}) - g \sum_n (X_n + \mathrm{h.c.})
        
            with

            Z = diag(1, w, w^2, ..., w^{d-1})  with w = e^{2\pi i / d}\n
            X = [[0  1  0  0  ...  0]
                 [0  0  1  0  ...  0]\n
                 [0  0  0  1  ...  0]\n
                 ...\n
                 [0  0  0  0  ...  1]\n
                 [1  0  0  0  ...  0]]

        Input:
            - backend
            - L: length of chain
            - d: local Hilbert space dimention
            - g: transverse field
            - J: coupling term
        """
        self.BE = backend
        
        self.d = d
        self.L = L
    
        # Model properties
        self.properties = [g, J]

        self.TE_MPO = MPO_TE_ClockModel_NN(backend=backend, L=L, d=d, properties=self.properties)
        self.MPS = MPS(backend=backend, L=L, d=d)

    def init_MPS(self, key='Z'):
        '''
        Initiates MPS.

        Input keys:
            - "X": groundstate for g-> infinity
            - "Z": groundstate for g=0
        '''
        if key == 'X':
            # X=1 (groundstate for g-> infinity)
            state = self.BE.ones((self.d,)) / self.BE.sqrt(self.BE.asarray(self.d))
        elif key == 'Z':
            # Z=1 (groundstate for g=0)
            state = self.BE.zeros((self.d,))
            state[0] = 1.
        B = state[None, :, None]
        S = self.BE.ones((1,), dtype=float)
        B_list, sv_list = [], []
        for i in range(self.L):
            B_list.append(self.BE.copy(B)) # legs [vL, p, vR]
            sv_list.append(self.BE.copy(S))
        sv_list.append(self.BE.copy(S))

        self.MPS.B_list = B_list
        self.MPS.sv_list = sv_list
        return B_list, sv_list



##########################################################
#### Clock Model (nearest- and next-nearest-Neighbor) ####
##########################################################

class ClockModel_NNN: 
    def __init__(self, backend: object, L:int, d: int, g: float, J1:float, J2:float):
        """
        Clock Model for nearest- and next-nearest-neighbors with the Hamiltonian

            H = -J1 \sum_n ( Z_n  Z_{n+1}^\dagger+ \mathrm{h.c.}) - J2 \sum_n (Z_n  Z_{n+2}^\dagger+  \mathrm{h.c.}) - g \sum_n (X_n + \mathrm{h.c.}) 

            with

            Z = diag(1, w, w^2, ..., w^{d-1})  with w = e^{2\pi i / d}\n
            X = [[0  1  0  0  ...  0]
                 [0  0  1  0  ...  0]\n
                 [0  0  0  1  ...  0]\n
                 ...\n
                 [0  0  0  0  ...  1]\n
                 [1  0  0  0  ...  0]]
                 
        Input:
            - backend
            - L: length of chain
            - d: local Hilbert space dimention
            - g: transverse field
            - J1: coupling term for nearest-neighbor
            - J2: coupling term for next-nearest-neighbor
        """
        self.BE = backend
        
        self.d = d
        self.L = L
    
        # Model properties
        self.properties = [g, J1, J2]

        self.TE_MPO = MPO_TE_ClockModel_NNN(backend=backend, L=L, d=d, properties=self.properties)
        self.MPS = MPS(backend=backend, L=L, d=d)

    def init_MPS(self, key='Z'):
        '''
        Initiates MPS according to key.
        
        Input:
            - key: either:
                - "X": groundstate for g-> infinity
                - "Z": groundstate for g=0
        '''

        if key == 'X':
            state = self.BE.ones((self.d,)) / self.BE.sqrt(self.BE.asarray(self.d))
            B = state[None, :, None]
            S = self.BE.ones((1,), dtype=float)
            B_list, sv_list = [], []
            for i in range(self.L):
                B_list.append(self.BE.copy(B)) # legs [vL, p, vR]
                sv_list.append(self.BE.copy(S))
            sv_list.append(self.BE.copy(S))

        elif key == 'Z':
            state = self.BE.zeros((self.d,))
            state[0] = 1.
            B = state[None, :, None]
            S = self.BE.ones((1,), dtype=float)
            B_list, sv_list = [], []
            for i in range(self.L):
                B_list.append(self.BE.copy(B)) # legs [vL, p, vR]
                sv_list.append(self.BE.copy(S))
            sv_list.append(self.BE.copy(S))

        elif key == 'Zcm':
            state_p = self.BE.zeros((self.d,))
            state_p[0] = 1.
            state_m = self.BE.zeros((self.d,))
            state_m[self.d//2] = 1.
            B_p = state_p[None, :, None]
            B_m = state_m[None, :, None]
            S = self.BE.ones((1,), dtype=float)
            B_list, sv_list = [], []
            for i in range(self.L):
                if i != self.L//2:
                    B_list.append(self.BE.copy(B_p)) # legs [vL, p, vR]
                    sv_list.append(self.BE.copy(S))
                else:
                    B_list.append(self.BE.copy(B_m)) # legs [vL, p, vR]
                    sv_list.append(self.BE.copy(S))
            sv_list.append(self.BE.copy(S))
        
        self.MPS.B_list = B_list
        self.MPS.sv_list = sv_list
        return B_list, sv_list
