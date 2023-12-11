
###########################################
############# Object Template #############
###########################################

class MPO_template:
    def __init__(self, backend: object, L:int, d: int, properties: list):
        
        # Must have, leave it as it is
        self.BE = backend

        # Must have, leave it as it is, but change self.D input
        self.L = L
        self.d = d
        self.D = 3

        # Must have, leave it as it is
        self.W_list = [] # legs: wL, wR, p, p* (dim: D, D, d, d)
        self.wL = None # legs: wR
        self.wR = None # legs: wL

        # MPO properties (No must have)
        self.p_1 = properties[0]
        self.p_2 = properties[1]
        self.P_3 = properties[2]

    # If the MPO is a time evolution operator U(t), the following function is a must have:
    def init_MPO(self, dt: float):

        # Initiate operator and left and right boundary vector
        wL = None # legs: wR
        w_list = None # legs: wL, wR, p, p*
        wR = None # legs: wL

        # Must have (write to "self" variables)
        self.W_list = w_list
        self.wL = wL
        self.wR = wR

        return wL, w_list, wR



###########################################
######## TE-MPO for Clock Model NN ########
###########################################

class MPO_TE_ClockModel_NN:
    def __init__(self, backend: object, L:int, d: int, properties: list):
        
        self.BE = backend

        self.L = L
        self.d = d
        self.D = 3

        self.W_list = []
        self.wL = None
        self.wR = None

        # MPO properties
        self.g = properties[0]
        self.J = properties[1]

        self.init_operators()
    
    def init_operators(self):
        self.Id = self.BE.eye(self.d)

        Z_dia = []
        for d_i in range(self.d):
            Z_dia.append(self.BE.exp(self.BE.asarray(2.j*self.BE.pi()*d_i/self.d))) 
        self.Z = self.BE.diag(Z_dia) # [p, p*]
        self.Z_dag = self.BE.dagger(self.Z)

        X = self.BE.eye(self.d, k=1)
        X[-1, 0] = 1.

        self.X = X  # [p, p*]
        self.X_dag = self.BE.dagger(X)
        self.XpX_dag = self.X + self.X_dag

    def init_MPO(self, dt: float):
        '''
        Create W^I approximation of U=exp(-i*dt*H).

        Input:
            - dt: time step
        
        Return:
            - wL
            - w_list
            - wR
        '''
        # Legs [wL, wR, p, p*]
        
        delta_t = -1.j * dt
        w = self.BE.zeros((self.D, self.D, self.d, self.d))
        w[0,0], w[0,1], w[0,2] = self.Id - delta_t * self.g * self.XpX_dag, self.Z, self.Z_dag
        w[1,0] = -delta_t * self.J* self.Z_dag
        w[2,0] = -delta_t * self.J* self.Z

        w_list = []
        for i in range(self.L):
            w_list.append(self.BE.copy(w))

        wL = self.BE.zeros(self.D)
        wL[0] = 1.
        wR = self.BE.copy(wL)

        self.W_list = w_list
        self.wL = wL
        self.wR = wR

        return wL, w_list, wR



###########################################
######## TE-MPO for Clock Model NNN #######
###########################################

class MPO_TE_ClockModel_NNN:
    def __init__(self, backend: object, L:int, d: int, properties: list):
        
        self.BE = backend

        self.L = L
        self.d = d
        self.D = 5

        self.W_list = []
        self.wL = None
        self.wR = None

        # MPO properties
        self.g = properties[0]
        self.J1 = properties[1]
        self.J2 = properties[2]

        self.init_operators()
    
    def init_operators(self):
        self.Id = self.BE.eye(self.d)

        Z_dia = []
        for d_i in range(self.d):
            Z_dia.append(self.BE.exp(self.BE.asarray(2.j*self.BE.pi()*d_i/self.d))) 
        self.Z = self.BE.diag(Z_dia) # [p, p*]
        self.Z_dag = self.BE.dagger(self.Z)

        X = self.BE.eye(self.d, k=1)
        X[-1, 0] = 1.

        self.X = X  # [p, p*]
        self.X_dag = self.BE.dagger(X)
        self.XpX_dag = self.X + self.X_dag

    def init_MPO(self, dt: float):
        '''
        Create W^I approximation of U=exp(-i*dt*H).

        Input:
            - dt: time step
        
        Return:
            - wL
            - w_list
            - wR
        '''
        # Legs [wL, wR, p, p*]
        
        delta_t = -1.j * dt
        w = self.BE.zeros((self.D, self.D, self.d, self.d))
        w[0,0], w[0,1], w[0,2] = self.Id - delta_t * self.g * self.XpX_dag, self.Z, self.Z_dag
        w[1,0], w[1,3] = -delta_t * self.J1* self.Z_dag, self.Id
        w[2,0], w[2,4] = -delta_t * self.J1* self.Z, self.Id
        w[3,0] = -delta_t * self.J2* self.Z_dag
        w[4,0] = -delta_t * self.J2* self.Z

        w_list = []
        for i in range(self.L):
            w_list.append(self.BE.copy(w))

        wL = self.BE.zeros(self.D)
        wL[0] = 1.
        wR = self.BE.copy(wL)

        self.W_list = w_list
        self.wL = wL
        self.wR = wR

        return wL, w_list, wR
