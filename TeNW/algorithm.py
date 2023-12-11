
###########################################
################# Imports #################
###########################################

from .mps_network import MPS
from math import sqrt



###########################################
######## MPO Time Evolution Engine ########
###########################################

class time_evolution_engine:
    def __init__(self, model: object):
        '''
        Time evolution engine. 

        Input: 
            - model
        
        Note model object can be obtained by "model.modelname()".
        '''
        self.mod = model
        self.BE = model.BE

    def check_options(self, options):
        options_default = {
            'N_sweeps': 1,
            'trunc_threshold': 1e-10,
            'chi_max': 100,
            'compute_error': False,
            'save_theta': False,
            'compute_distance': False
            }
        if options is None:
            options = options_default
        else:
            for key in options_default:
                if key not in options:
                    options[key] = options_default[key]
        return options
    
    # Helper Functions
    ##################

    def add_to_R(self, R, B, W, B_tp1):
        R = self.BE.tensordot( B, R, axes=([2],[0]) ) # v_L, p, w_L, v_L'
        R = self.BE.tensordot( W, R, axes=([1,2],[2,1]) ) # w_L, bra_n, v_L, v_L'
        R = self.BE.tensordot( self.BE.conj(B_tp1), R, axes=([1,2],[1,3]) ) # v_L', w_L, v_L
        R = self.BE.transpose( R, perm=[2,1,0] )
        return R
    
    def init_L_list(self):
        '''
        Output legs: v_R, w_R, v_R'
        '''
        L = self.BE.zeros([1,self.mod.TE_MPO.D,1]) # v_R, w_R, v_R'
        L[0,:,0] = self.mod.TE_MPO.wL
        L_list = [L]
        return L_list

    def init_R_list(self):
        '''
        Output legs: v_L, w_L, v_L'
        '''
        R = self.BE.zeros([1,self.mod.TE_MPO.D,1]) # v_L, w_L, v_L'
        R[0,:,0] = self.mod.TE_MPO.wR
        R_list = [R]
        return R_list
        
    def get_R_list(self, B_list, B_list_tp1):
        '''
        Output legs: v_L, w_L, v_L'
        '''
        R_list = self.init_R_list()
        R = R_list[0]
        W_list = self.mod.TE_MPO.W_list
        for i in range(self.mod.L-2):
            n =(self.mod.L-1)-i
            R = self.add_to_R(R, B_list[n], W_list[n], B_list_tp1[n])
            R_list.append( R )
        return R_list
        
    def get_theta_new(self, n, L, R, B_list, W_list):
        '''
        Input legs: 
            L_list: v_R, w_R, v_R'
            R_list: v_L, w_L, v_L'
            B_list: v_L, p, v_R
            W_list: w_L, w_R, ket_n, bra_n
        Output legs: 
            L_new: v_R', v_R, w_R, bra_n
            R_new: v_L' v_L, w_L, bra_np1
            theta_new:v_L, bra_n, bra_np1, v_R
        '''
        L_new = self.BE.tensordot( L, B_list[n], axes=([0],[0]) )  # w_R, v_R', p, v_R
        L_new = self.BE.tensordot( L_new, W_list[n], axes=([0,2],[0,2]) )  # v_R', v_R, w_R, bra_n
        R_new = self.BE.tensordot( R, B_list[n+1], axes=([0],[2]) )  # w_L, v_L' v_L, p_p1
        R_new = self.BE.tensordot( R_new, W_list[n+1], axes=([0,3],[1,2]) )  # v_L' v_L, w_L, bra_np1
        theta_new = self.BE.tensordot( L_new, R_new, axes=([1,2],[1,2]) )  # v_R', bra_n, v_L', bra_np1
        theta_new = self.BE.transpose( theta_new , perm=(0,1,3,2)) # v_R', bra_n, bra_np1, v_L' = v_L, bra_n, bra_np1, v_R
        return L_new, R_new, theta_new


    # Time Step (update function)
    #############################

    def evolve_time_step(self, truncator, options: dict=None):
        '''
        Perform para:N_sweeps from (left -> right) and back (left <- right) and obtain updated MPS.
        Before performing a time evolution create the time evolution MPO with "model.TE_MPO.init_MPO()".
        Note that the updated MPS is not a return value of this function. Access the updated MPS via "model.MPS".

        Inputs:
            - truncator object
            - options: dictionary with keys:
                - N_sweeps: int
                - trunc_threshold: float
                - chi_max: int
                - compute_error: bool
                - save_theta: bool
                - compute_distance: bool
        
        Note truncator object can be obtained by "truncator.initiate_truncator()".

        Return:
            - results: Dictionary with key's:
                - truncation_error
                - norm_deviation
                - distance
                - theta_middle
                - total_walltime
                - trunc_walltime
        '''
        t_tot, T = self.BE.start_stop_time()

        options = self.check_options(options)
        N_sweeps = options['N_sweeps']
        trunc_threshold = options['trunc_threshold']
        chi_max = options['chi_max']
        compute_error = options['compute_error']
        save_theta = options['save_theta']
        compute_distance = options['compute_distance']
        t_tr = []
    
        distance_list = []
        theta_mid = None

        # Guess MPS of next timestep t+delta_t: choose MPS from t
        R_list = self.get_R_list(self.mod.MPS.B_list, self.mod.MPS.B_list) # v_L, w_L, v_L'

        # Compute Distance at beginning
        if compute_distance:
            norm_U_MPS = sqrt(self.mod.MPS.get_norm_sq_of_U_MPS(self.mod.TE_MPO).real)
            overlap = self.mod.MPS.get_expVal_of_MPO(self.mod.TE_MPO).real
            N_MPS_up = sqrt(self.mod.MPS.get_norm_sq().real)
            distance_list.append( sqrt(abs(2-2*overlap.real/(norm_U_MPS*N_MPS_up)))  )

        # Sweep through the sites and calculate new value for the MPS
        for i in range(N_sweeps):
            trunc_err_list = []
            norm_err_list = []
            
            # left -> right
            ###############
            
            # Initialize left side
            L_list = self.init_L_list()
            
            for n in range(self.mod.L-2):
                m = (self.mod.L-2)-n

                # Calculate new theta (tp1)
                L_new, R_new, theta_new = self.get_theta_new( n, L_list[n], R_list[m], self.mod.MPS.B_list, self.mod.TE_MPO.W_list) # v_R', v_R, w_R, bra_n / v_L', bra_n, bra_np1, v_R'

                # Decompose
                t_tr0, T = self.BE.start_stop_time()
                A_n, sv_np1, X_np1, trunc_err, N = truncator.decompose(theta_new, chi_max, trunc_threshold, left_side=True, comp_error=compute_error)
                t_tr0, T = self.BE.start_stop_time(t_tr0)
                t_tr.append(T)

                # Update left side with new A_n
                L_new = self.BE.tensordot( L_new, self.BE.conj(A_n), axes=([0,3],[0,1]) ) # v_R, w_R, v_R'
                L_list.append( L_new )
                
            # left <- right
            ###############
            
            # Initialize storage lists
            B_list_tp1 = []
            sv_list_tp1 = [self.BE.asarray([1])]

            # Initialize right side
            R_list = self.init_R_list()
            
            for m in range(self.mod.L-1):
                n = (self.mod.L-2)-m

                # Calculate new theta (tp1)
                L_new, R_new, theta_new = self.get_theta_new( n, L_list[n], R_list[m], self.mod.MPS.B_list, self.mod.TE_MPO.W_list) # v_R', v_R, w_R, bra_n / v_L', bra_n, bra_np1, v_R'
                
                # Save theta if needed
                if save_theta:
                    if n == (self.mod.L-2)//2:
                        theta_mid = self.BE.copy(theta_new)

                # Decompose
                if m != self.mod.L-2: compute_C = False
                else: compute_C = True
                t_tr0, T = self.BE.start_stop_time()
                X_n, sv_np1, B_np1, trunc_err, N = truncator.decompose(theta_new, chi_max, trunc_threshold, left_side=False, comp_error=compute_error, compute_C=compute_C)
                t_tr0, T = self.BE.start_stop_time(t_tr0)
                t_tr.append(T)
                trunc_err_list.append(trunc_err)
                norm_err_list.append(1-N)
                
                # Save B Tensors and schmidt values
                sv_list_tp1.append(sv_np1)
                B_list_tp1.append(B_np1)
                
                # Update right side with new B_np1
                R_new = self.BE.tensordot( R_new, self.BE.conj(B_np1), axes=([0,3],[2,1]) ) # v_L, w_L, v_L'
                R_list.append( R_new )

            B_0 = X_n # last X_0 = C_0 = B_0 (since first schmidt value [1])
            B_list_tp1.append(B_0) 
            sv_list_tp1.append(self.BE.asarray([1]))

            # flip since we appended from other side
            B_list_tp1 = B_list_tp1[::-1]
            sv_list_tp1 = sv_list_tp1[::-1]

            # Compute Distance
            if compute_distance:
                overlap = self.BE.tensordot( R_new, self.mod.MPS.B_list[0], axes=([0],[2]) )  # w_L, v_L' v_L, p
                overlap = self.BE.tensordot( overlap, self.mod.TE_MPO.W_list[0], axes=([0,3],[1,2]) )  # v_L' v_L, w_L, p'
                overlap = self.BE.tensordot( overlap, self.BE.conj(B_0), axes=([0,3],[2,1]) ) # v_L, w_L, v_L'
                overlap = self.BE.tensordot( overlap, self.mod.TE_MPO.wL, axes=([1],[0]) ) # v_L, v_L'
                overlap = self.BE.trace(overlap)

                MPS_swipe = MPS(self.mod.BE, self.mod.L, self.mod.d) # Do not overwrite "self.mod.MPS.B_list", since values are still needed in swipes
                MPS_swipe.B_list = B_list_tp1
                MPS_swipe.sv_list = sv_list_tp1
                N_MPS_up = sqrt(MPS_swipe.get_norm_sq().real)
                distance_list.append( sqrt(abs(2-2*overlap.real/(norm_U_MPS*N_MPS_up))) )
                
                MPS_swipe.B_list = None 
                MPS_swipe.sv_list = None
        # Overwrite initial MPS with new values
        self.mod.MPS.B_list  = B_list_tp1
        self.mod.MPS.sv_list = sv_list_tp1

        t_tot, T = self.BE.start_stop_time(t_tot)

        results = {
            'truncation_error':trunc_err_list,
            'norm_deviation': norm_err_list,
            'distance': distance_list,
            'theta_middle': theta_mid,
            'total_walltime': T,
            'trunc_walltime': t_tr,
        }
        return results
