
###########################################
################# Imports #################
###########################################

# Packages
import numpy as np
import pickle as pk
import os
import sys
import datetime

###########################################
############### MPS Object ################
###########################################

class MPS:
    def __init__(self, backend, L: int, d: int):

        self.BE = backend

        self.L       = L
        self.d       = d

        self.B_list  = None
        self.sv_list = None
    

    # Generate MPS from psi list
    ############################
        
    def get_MPS(self, psi: list, chi_max: int, M_type: str='A', QR_method=False):
        '''
        Input:
            state: list with d^L entries with amplitude for corresponding state
            chi_max: max chi allowed
            M_type: ist either 'A' or 'B'
        ToDo: Implement SVD from the other side, that the 'B' form can be calcualted without inverting schmidt values
        '''
        if len(psi) != self.d**(self.L):
            raise ValueError(f'State psi has {len(psi)} entries instead of {self.d**(self.L)}')
        if M_type != 'A' and M_type != 'B':
            raise ValueError(f'M_type "{M_type}" not implemented. Try A or B')

        M_list = []
        sv_list = []
        if not QR_method: sv_list.append(self.BE.asarray([1.]))

        psi   = self.BE.asarray(psi)
        psi_R = self.BE.reshape(psi, (1, self.d**(self.L)))
        
        for n in range(self.L):
            chi_n, dim_R = self.BE.shape( psi_R )
            psi_R_new    = self.BE.reshape(psi_R, (chi_n*self.d, dim_R//self.d))

            if QR_method and n==self.L-1:
                X_n, psi_tilde = self.BE.qr(psi_R_new)
            else:
                X_n, lambda_np1, psi_tilde = self.BE.svd(psi_R_new)
            
            chivC = chi_max
        
            X_n       = X_n[:, :chivC]
            psi_tilde = psi_tilde[:chivC, :]
            chi_new = self.BE.shape(X_n)[1]
            
            X_n   = self.BE.reshape(X_n, (chi_n, self.d, chi_new) )

            if QR_method and n==self.L-1:
                psi_R = psi_tilde
            else:
                psi_R = self.BE.tensordot( self.BE.diag(lambda_np1), psi_tilde, axes=([1],[0]) )

            if not QR_method:
                lambda_np1 = lambda_np1[:chivC]
                N          = self.BE.norm(lambda_np1) 
                lambda_np1 = lambda_np1 / N

            if M_type == 'B':
                sv_inv = self.BE.diag(sv_list[-1]**(-1))
                X_n = self.BE.tensordot(sv_inv, self.BE.tensordot( X_n, self.BE.diag(lambda_np1), axes=([2],[0]) ), axes = ([1],[0]))
            M_list.append(X_n)

            if not QR_method: sv_list.append(lambda_np1)
        assert self.BE.shape(psi_R) == (1, 1)
        
        self.B_list = M_list
        self.sv_list = sv_list
        return M_list, sv_list
    

    # Generate psi list from MPS
    ############################

    def compute_MPS(self):
        '''
        This function contracts the MPS.
        Warning: This is only possible for small MPS. Otherwise memory will run out.
        '''
        contr = self.B_list[0]
        for i in range(1, self.L):
            contr = self.BE.tensordot(contr, self.B_list[i], axes=([-1],[0]))
        contr = self.BE.reshape(contr, (int(self.d**self.L)))
        return contr


    # MPS information functions
    ###########################

    def get_bonds(self):
        '''
        Calculate the bond dimension between sites n and n+1.

        Return:
            - bond_list: list containing the bond dimensions
        '''
        bond_list = []
        for i in range(1,len(self.B_list)):
            B = self.B_list[i]
            chi_l = self.BE.shape(B)[0]
            bond_list.append(int(chi_l))
        return bond_list
    
    def get_vN_entropy(self):
        '''
        Calculate the Von Neuman Entropy defined as

            S_\mathrm{vN} = - \sum_{\\alpha} \Lambda^2_\\alpha \mathrm{log} \Lambda^2_\\alpha

        between the bipartite system cut between site n and n+1.

        Return:
            - vN_entropy_list: list containing the Von Neuman Entropys
        '''
        vN_entropy_list = []
        for i in range(1,len(self.sv_list)):
            sv = self.sv_list[i]
            if sv is None:
                vN_entropy_list = None
                break
            else:
                sv = self.BE.real(sv)
                sv = sv[sv > 1e-10] ** 2
                vN_entropy_list.append(-self.BE.dot(self.BE.log(sv), sv))
        return vN_entropy_list

    def get_norm_sq(self):
        N = self.BE.tensordot(self.BE.conj(self.B_list[0]), self.B_list[0],axes=([0,1],[0,1])) # v_R, v_R'
        for n in range(1, len(self.B_list)):
            N = self.BE.tensordot(N, self.B_list[n],axes=([0],[0])) # v_R', p, v_R
            N = self.BE.tensordot(N, self.BE.conj(self.B_list[n]),axes=([0,1],[0,1])) # v_R, v_R'
        N = self.BE.trace(N)
        return N

    def get_norm_sq_of_U_MPS(self, MPO: object):
        '''
        Get <MPS|U^dagger U|MPS>
        
        return: scalar
        '''

        B_list = self.B_list
        W_list = MPO.W_list

        # Start at first site (note that the first schmidt values are ignored since they are [1])
        contr1 = self.BE.tensordot( MPO.wL, W_list[0], axes=([0],[0]) ) # w_R1, p, p*
        contr2 = self.BE.tensordot( self.BE.conj(MPO.wL), self.BE.conj(W_list[0]), axes=([0],[0]) ) # w_R2, p, p*
        contr = self.BE.tensordot( contr1, contr2, axes=([2],[2]) ) # w_R1, p, w_R2, p*
        contr = self.BE.tensordot( B_list[0], contr, axes=([1],[1]) ) # v_L, v_R, w_R1, w_R2, p*
        contr = self.BE.tensordot( contr, self.BE.conj(B_list[0]), axes=([0,4],[0,1]) ) # v_R, w_R1, w_R2, v_R'
        contr = self.BE.transpose(contr ,perm=(1,2,0,3)) # w_R1, w_R2, v_R, v_R'

        # Loop through sites
        for n in range(1,self.L):
            contr = self.BE.tensordot(contr, W_list[n], axes=([0],[0])) #  w_R2, v_R, v_R', w_R, p, p*
            contr = self.BE.tensordot(contr, self.BE.conj(W_list[n]), axes=([0,5],[0,3])) # v_R, v_R', w_R1, p, w_R2, p*
            contr = self.BE.tensordot(contr, B_list[n], axes=([0,3],[0,1])) # v_R', w_R1, w_R2, p*, v_R
            contr = self.BE.tensordot(contr, self.BE.conj(B_list[n]), axes=([0,3],[0,1])) # w_R1, w_R2, v_R, v_R'
        
        # End at last site
        contr = self.BE.tensordot( contr, MPO.wR, axes=([0],[0]) ) # w_R2, v_R, v_R'
        contr = self.BE.tensordot( contr, self.BE.conj(MPO.wR), axes=([0],[0]) ) # v_R, v_R'
        N = self.BE.trace(contr)

        return N
    
    def get_overlap(self, B_list: list):
        '''
        Get overlap of MPS with a B_list (legs: vL, p, vR)
        '''
        contr = self.BE.asarray([[1]])
        for n in range(len(B_list)):
            contr = self.BE.tensordot(contr, self.B_list[n],axes=([0],[0])) # v_R', p, v_R
            contr = self.BE.tensordot(contr, self.BE.conj(B_list[n]),axes=([0,1],[0,1])) # v_R, v_R'
        return self.BE.trace(contr)
    
    def get_expVal_of_MPO(self, MPO: object, lim_for_imag: float = 1e-10, hide_warning:bool=True):
        '''
        Get expectation value of MPO
        
        return: scalar
        '''

        B_list = self.B_list
        W_list = MPO.W_list

        # Start at first site (note that the first schmidt values are ignored since they should be [1])
        contr = self.BE.tensordot( MPO.wL, W_list[0], axes=([0],[0]) ) # w_R, p, p*
        contr = self.BE.tensordot( B_list[0], contr, axes=([1],[1]) ) # v_L, v_R, w_R, p*
        contr = self.BE.tensordot( contr, self.BE.conj(B_list[0]), axes=([0,3],[0,1]) ) # v_R, w_R, v_R'
        contr = self.BE.transpose(contr ,perm=(1,0,2)) # w_R, v_R, v_R'

        # Loop through sites
        for n in range(1,self.L):
            contr = self.BE.tensordot(contr, W_list[n], axes=([0],[0])) # v_R , v_R', w_R, p, P*
            contr = self.BE.tensordot(contr, B_list[n], axes=([0,3],[0,1])) # v_R', w_R, p*, v_R
            contr = self.BE.tensordot(contr, self.BE.conj(B_list[n]), axes=([0,2],[0,1])) # w_R, v_R, v_R'
        
        # End at last site
        contr = self.BE.tensordot( contr, MPO.wR, axes=([0],[0]) ) # v_R, v_R'
        exp_value = self.BE.trace(contr)

        if exp_value.imag > lim_for_imag and not hide_warning:
            print("Imaginary part of expectation value of MPO is "+str(exp_value.imag))
        
        return exp_value
    
    
    def get_expVal_of_Op(self, operator, sv_available:bool):
        '''
        Get expectation value of local operator for each site.
        
        Input:
            - operator: with legs: p, p*
            - sv_available: If the MPS has Schmidt values or not. The 'SVD' and 'QR+CBE' method 
                            calculates the Schmidt values (sv_available=True),
                            the 'QR' methode does not (sv_available=False)

        Return:
            - exp_value_list: list containing the expectation values of the local operator for each site
        '''
        lim_for_imag = 1e-10
        hide_warning =True

        B_list = self.B_list
        
        exp_value_list = []
        if sv_available:
            sv_list = self.sv_list
            for n in range(self.L):
                thata = self.BE.tensordot( self.BE.diag(sv_list[n]), B_list[n], axes=([1],[0])) # vL, vR | vL, p, vR -> vL, p, vR 
                exp_value = self.BE.tensordot( operator, thata, axes=([0],[1])) # p, p* | vL, p, vR -> p*, vL, vR
                exp_value = self.BE.tensordot( exp_value, self.BE.conj(thata), axes=([0,1,2],[1,0,2])) # p*, vL, vR | vL, p, vR -> []
                if exp_value.imag > lim_for_imag and not hide_warning:
                    print("Warning: Imaginary part of expectation value of local operator is "+str(exp_value.imag))
                exp_value_list.append(exp_value)
        else:
            L = self.BE.tensordot( B_list[0], self.BE.conj(B_list[0]), axes=([0],[0])) # p, vR, p*, vR'
            for n in range(1, self.L):
                exp_value = self.BE.tensordot( L, operator, axes=([0,2],[0,1])) # p, vR, p*, vR' | p, p* -> vR, vR'
                exp_value = self.BE.trace(exp_value)
                if exp_value.imag > lim_for_imag and not hide_warning:
                    print("Warning: Imaginary part of expectation value of local operator is "+str(exp_value.imag))
                exp_value_list.append(exp_value)
                if n != self.L-1:
                    L = self.BE.tensordot( L, self.BE.eye(self.BE.shape(L)[0]), axes=([0,2],[0,1])) # p, vR, p*, vR' | p, p* -> vR, vR'
                    L = self.BE.tensordot( L, B_list[n], axes=([0],[0])) # vR, vR' | vL, p, vR -> vR', p, vR
                    L = self.BE.tensordot( L, self.BE.conj(B_list[n]), axes=([0],[0])) # vR', p, vR | vL', p*, vR' -> p, vR, p*, vR'

        return exp_value_list
    

    # GPU <-> CPU converter
    ###########################################

    def convert_to_backend(self, backend: object):
        self.convert_MPS_to_CPU()
        for i in range(len(self.B_list)):
            self.B_list[i] = backend.asarray(self.B_list[i])
        for i in range(len(self.sv_list)):
            sv = self.sv_list[i]
            if not isinstance(sv, np.ndarray):
                sv = [sv]
            self.sv_list[i] = backend.asarray(sv)
        self.BE = backend

    def convert_to_CPU(self):
        B_list = []
        sv_list = []
        for i in range(len(self.B_list)):
            B_list.append( self.BE.copy_to_np_CPU(self.B_list[i]) )
        for i in range(len(self.sv_list)):
            sv_list.append( self.BE.copy_to_np_CPU(self.sv_list[i]) )

        return B_list, sv_list

    def convert_MPS_to_CPU(self):
        B_list, sv_list = self.convert_to_CPU()
        self.B_list = B_list
        self.sv_list = sv_list


    # Functions to save and load MPS
    ###########################################

    def save_MPS(self, file_name:str, subdir=None, with_timestamp:bool=False):
        '''
        Save the MPS object.

        Inputs:
            - file_name: The name you want the file to have, for example 'my_MPS'
            - subdir: For example '/Output/Data/'. Choose None if you want to save in the current directory
            - with_timestamp: If True the date and time will be added before the file_name. 
        '''
        if subdir is None:
            dir = ''
        else:
            dir = os.path.dirname(os.path.abspath(sys.argv[0]))+subdir
        
        if with_timestamp:
            now = datetime.datetime.now()
            now_string = now.strftime("%Y_%m_%d_%H%M%S")+'_'
        else:
            now_string = ''

        file_name_new = dir+now_string+file_name
        file_name = file_name_new+'.txt'

        B_list, sv_list = self.convert_to_CPU()
        data = [self.BE, self.L, self.d, B_list, sv_list]
        
        file = open(file_name,'wb')
        file.write(pk.dumps(data))
        file.close()

        return file_name_new

    def load_MPS(self, file_name:str, subdir=None):
        '''
        Load a MPS object.
        Note: Current MPS will be overwritten!
        
        Inputs:
            - file_name: Name of the file you want to load without
            - subdir: Sub directory of file, for example '/Output/Data/'. Choose None if it is in the current directory
        '''

        if subdir is None:
            dir = ''
        else:
            dir = os.path.dirname(os.path.abspath(sys.argv[0]))+subdir

        file_name = dir+file_name+'.txt'
        
        file = open(file_name,'rb')
        dataPickle = file.read()
        file.close()
        data = pk.loads(dataPickle)

        _, self.L, self.d, self.B_list, self.sv_list = data
        self.convert_to_backend(self.BE)
