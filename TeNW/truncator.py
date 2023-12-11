
###########################################
######### Initialisation Function #########
###########################################

def initiate_truncator(backend: object, truncator: str):
    '''
    Initiate truncator.
    
    Input:
        - backend
        - truncator: 'SVD' or 'QR' or 'QR+CBE'

    Note backend object can be obtained by "backend.initiate_backend()".

    Return:
        - truncator (object)
    '''
    TC = None
    if truncator == 'SVD' or truncator =='svd':
        TC = svd(backend)
    elif truncator == 'QR' or truncator =='qr':
        TC = qr(backend)
    elif truncator == 'QR+CBE' or truncator =='qr+cbe':
        TC = qr_cbe(backend)
    else:
        raise ValueError(f'Truncator "{truncator}" not implemented. Try: SVD, QR, QR+CBE')
    return TC



###########################################
############## SVD Truncator ##############
###########################################

class svd:
    def __init__(self, BE):
        self.BE = BE
        self.name = 'SVD'

    def decompose(self, theta, chi_max: int, threshold:float=1e-10, left_side:bool=False, comp_error:bool=False, compute_C:bool=False):
        '''

        '''
        trunc_err = None
        
        # SVD
        chi_l, d1, d2, chi_r = self.BE.shape(theta) # v_L, p_n, p_np1, v_R
        theta_resh = self.BE.reshape(theta, (chi_l*d1, d2*chi_r))
        X_n, S, X_np1  = self.BE.svd(theta_resh)
        
        # Truncate
        chi_cbe  = min(chi_max, self.BE.sum(S > threshold))
        A_n    = X_n[:, :chi_cbe]
        sv_np1 = S[:chi_cbe]
        B_np1  = X_np1[:chi_cbe, :]
        
        # Reshape
        A_n   = self.BE.reshape(A_n, (chi_l, d1, chi_cbe))
        B_np1 = self.BE.reshape(B_np1, (chi_cbe, d2, chi_r))

        # Error 
        if comp_error:
            trunc_err = self.BE.norm(S[chi_cbe:])/self.BE.norm(S)
        
        # Normalisation 
        N = self.BE.norm(sv_np1)
        sv_np1 = sv_np1 / N

        if left_side:
            X_n = A_n
            X_np1 = self.BE.tensordot( self.BE.diag(sv_np1), B_np1, axes=([1],[0]) )
        else:
            X_n = self.BE.tensordot( A_n, self.BE.diag(sv_np1), axes=([2],[0]) )
            X_np1 = B_np1

        return X_n, sv_np1, X_np1, trunc_err, N



###########################################
############## QR Truncator ###############
###########################################

class qr:
    def __init__(self, BE):
        self.BE = BE
        self.name = 'QR'
    
    def decompose(self, theta, chi_max: int, threshold:float=1e-10, left_side: bool=True, comp_error: bool=False, compute_C: bool =False):
        '''

        '''
        num_its = 1

        trunc_err = None
        
        chi_l, d1, d2, chi_r = self.BE.shape(theta)  # v_L, p_n, p_np1, v_R
        theta_resh = self.BE.reshape(theta, (chi_l*d1, d2*chi_r))
        chi = min(chi_l*d1, chi_max, d2*chi_r)

        if left_side:
            Q_n = theta_resh[:, :chi_max]
            for i in range(num_its):
                guess_R  = self.BE.tensordot( self.BE.conj(Q_n), theta_resh, axes=([0],[0])) # v_L, v_R (chi_new, d2*chi_r)
                L, Q_np1 = self.BE.lq( guess_R )
                guess_L = self.BE.tensordot( theta_resh, self.BE.conj(Q_np1), axes=([1],[1])) # v_L, v_R (chi_l*d1, chi_new)
                Q_n, R  = self.BE.qr( guess_L )
        else:
            Q_np1 = theta_resh[:chi_max, :]
            for i in range(num_its):
                guess_L = self.BE.tensordot( theta_resh, self.BE.conj(Q_np1), axes=([1],[1])) # v_L, v_R (chi_l*d1, chi_new)
                Q_n, R  = self.BE.qr( guess_L )
                guess_R  = self.BE.tensordot( self.BE.conj(Q_n), theta_resh, axes=([0],[0])) # v_L, v_R (chi_new, d2*chi_r)
                L, Q_np1 = self.BE.lq( guess_R )

        if left_side:
            # Get A_n and C_np1
            X_n = self.BE.reshape( Q_n, (chi_l, d1, chi) )
      
            if compute_C or comp_error:
                X_np1 = self.BE.tensordot(R, Q_np1, axes=([1],[0]))
                X_np1 = self.BE.reshape( X_np1, (chi, d2, chi_r) )
            else:
                X_np1 = None

        else:
            # Get C_n and B_np1
            X_np1 = self.BE.reshape( Q_np1, (chi, d2, chi_r) )
            
            if compute_C or comp_error:
                X_n = self.BE.tensordot(Q_n, L, axes=([1],[0]))
                X_n = self.BE.reshape( X_n, (chi_l, d1, chi) )
            else:
                X_n = None

        # Error 
        if comp_error:
            theta_appr = self.BE.tensordot(X_n, X_np1, axes=([2],[0]))
            trunc_err = self.BE.norm(theta-theta_appr)/self.BE.norm(theta)

        sv_np1 = None

        if left_side:
            N = self.BE.norm(R)
            if compute_C or comp_error:
                X_np1 = X_np1 / N # C_np1
        else:
            N = self.BE.norm(L)
            if compute_C or comp_error:
                X_n = X_n / N # C_n

        return X_n, sv_np1, X_np1, trunc_err, N

  

###########################################
############# QR+CBE Truncator #############
###########################################

class qr_cbe:
    def __init__(self, BE):
        self.BE = BE
        self.name = 'QR+CBE'
    
    def decompose(self, theta, chi_max: int, threshold:float=1e-10, left_side: bool=True, comp_error: bool=False, compute_C: bool =False):
        '''

        '''
        num_its = 1

        trunc_err = None

        chi_l, d1, d2, chi_r = self.BE.shape(theta)  # v_L, p_n, p_np1, v_R
        theta_resh = self.BE.reshape(theta, (chi_l*d1, d2*chi_r))

        #chi = min(chi_l*d1, chi_max, d2*chi_r)
        Q_np1 = theta_resh[:chi_max, :]
        
        for i in range(num_its):
            guess_L = self.BE.tensordot( theta_resh, self.BE.conj(Q_np1), axes=([1],[1])) # v_L, v_R (chi_l*d1, chi_new)
            Q_n, R  = self.BE.qr( guess_L )
            guess_R  = self.BE.tensordot( self.BE.conj(Q_n), theta_resh, axes=([0],[0])) # v_L, v_R (chi_new, d2*chi_r)
            L, Q_np1 = self.BE.lq( guess_R )

        r  = min(self.BE.shape(L)[0], self.BE.shape(L)[1]) 
        if left_side:
            # Get A_n, sv_np1 and C_np1
            Su_sq, U = self.BE.eig( self.BE.tensordot(L, self.BE.dagger(L), axes=([1],[0])) )
            Su_sq = self.BE.abs(Su_sq)
            order = self.BE.flip(self.BE.argsort(Su_sq))
            S     = self.BE.sqrt(Su_sq[order])[:r]
            U     = U[:, order][:,:r]

            X_n = self.BE.tensordot(Q_n, U, axes=([1],[0])) # A_n

            if compute_C or comp_error:
                X_np1 = self.BE.tensordot(L, Q_np1, axes=([1],[0]))
                X_np1 = self.BE.tensordot(self.BE.dagger(U), X_np1, axes=([1],[0])) # C_np1
            else:
                X_np1 = None # C_np1
            
            # Truncate
            chi_cbe  = min(chi_max, self.BE.sum(S > threshold))
            X_n    = self.BE.reshape( X_n[:, :chi_cbe], (chi_l, d1, chi_cbe) )
            sv_np1 = S[:chi_cbe]
            if compute_C or comp_error:
                X_np1 = self.BE.reshape( X_np1[:chi_cbe, :], (chi_cbe, d2, chi_r))

        else:
            # Get C_n, sv_np1 and B_np1
            Sv_sq, V = self.BE.eig( self.BE.tensordot(self.BE.dagger(L), L, axes=([1],[0])) )
            Sv_sq = self.BE.abs(Sv_sq)
            order = self.BE.flip(self.BE.argsort(Sv_sq))
            S     = self.BE.sqrt(Sv_sq[order])[:r]
            V_dag = self.BE.dagger(V)[order, :][:r,:]

            X_np1 = self.BE.tensordot(V_dag, Q_np1, axes=([1],[0])) # B_np1

            if compute_C or comp_error:
                X_n = self.BE.tensordot(Q_n, L, axes=([1],[0]))
                X_n = self.BE.tensordot(X_n, self.BE.dagger(V_dag), axes=([1],[0])) # C_n
            else:
                X_n = None # C_n
            
            # Truncate
            chi_cbe = min(chi_max, self.BE.sum(S > threshold))
            sv_np1 = S[:chi_cbe]
            X_np1 = self.BE.reshape( X_np1[:chi_cbe, :], (chi_cbe, d2, chi_r))
            if compute_C or comp_error:
                X_n = self.BE.reshape( X_n[:, :chi_cbe], (chi_l, d1, chi_cbe) )
                
        # Error 
        if comp_error:
            theta_appr = self.BE.tensordot(X_n, X_np1, axes=([2],[0]))
            trunc_err = self.BE.norm(theta-theta_appr)/self.BE.norm(theta)
        
        # Normalisation 
        N = self.BE.norm(sv_np1)
        sv_np1 = sv_np1 / N
        if compute_C or comp_error:
            if left_side:
                X_np1 = X_np1 / N # C_np1
            else:
                X_n = X_n / N # C_n
        
        return X_n, sv_np1, X_np1, trunc_err, N
