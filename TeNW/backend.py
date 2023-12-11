
###########################################
################# Imports #################
###########################################

# Packages
import numpy as np
try: import torch
except ModuleNotFoundError:
    print('Failed to import torch. Visit www.pytorch.org for more information about torch.\n')
import time


###########################################
######### Initialisation Function #########
###########################################

def initiate_backend(btype: str = 'numpy', device: str = 'cpu', dtype: str = 'complex128'):
    '''
    Initiate backend.
    
    Input:
        - btype: 'numpy' or 'torch'
        - device: 'cpu' or 'gpu' (or 'cuda')
        - dtype: 'complex128' or 'complex64'

    Return:
        - backend (object)
    '''
    BE = None
    if btype == 'numpy':
        BE = numpy_backend(dtype=dtype)
        if not (device == 'cpu' or device =='CPU'):
            raise ValueError(f'Backend with numpy only supports CPU. Use torch to run backend on GPU.')
    elif btype == 'torch':
        BE = torch_backend(dtype=dtype, device=device)
    else:
        raise ValueError(f'Backend "{btype}" not implemented. Try: numpy or torch')
    return BE



###########################################
############## Numpy Backend ##############
###########################################

class numpy_backend:
    def __init__(self, dtype: str='complex128'):
        if isinstance(dtype, str):
            if dtype == 'complex128':
                dtype = np.complex128
            elif dtype == 'complex64':
                dtype = np.complex64
            else:
                raise ValueError(f'Could not parse numpy dtype from "{dtype}". Try complex128 or complex64')
        self.dtype = dtype
        self.name = 'numpy'
        self.device = 'cpu'
    
    def start_stop_time(self, t0=0):
        t1 = time.perf_counter()
        return t1, t1-t0

    def asarray(self, a):
        return np.asarray(a, self.dtype)
    
    def tensordot(self, a, b, axes):
        return np.tensordot(a, b, axes)

    def transpose(self, a, perm=(1,0)):
        return np.transpose(a, perm)

    def conj(self, a):
        return np.conj(a)
    
    def dagger(self,a):
        return np.conj(a).T

    def reshape(self, a, shape):
        return np.reshape(a, shape)

    def svd(self, a, compute_uv=True):
        return np.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)
    
    def qr(self, a):
        return np.linalg.qr(a)
    
    def lq(self, a):
        q, r = np.linalg.qr(a.T)
        return r.T, q.T
    
    def norm(self, a):
        return np.linalg.norm(a)

    def norm_sq(self, a):
        return np.real(np.sum(a * np.conj(a)))

    def real(self, a):
        return np.real(a)

    def dot(self, a, b):
        return np.dot(a, b)
    
    def log(self, a):
        return np.log(a)
    
    def eye(self, n, m=None, k=0):
        return np.eye(n, M=m, k=k, dtype=self.dtype)
    
    def diag(self, a):
        return np.diag(a)
    
    def copy(self, a):
        return np.copy(a)
    
    def eig(self, a):
        return np.linalg.eig(a)
    
    def argsort(self, a):
        return np.argsort(a)
    
    def zeros(self, size):
        return np.zeros(size, dtype=self.dtype)
    
    def ones(self, size, dtype=None):
        if dtype is None: dtype = self.dtype
        return np.ones(size, dtype=dtype)
    
    def sum(self, a):
        return np.sum(a)
    
    def flip(self, a, axis=[0]):
        return np.flip(a, axis=axis)
    
    def trace(self, a):
        return np.trace(a)
    
    def abs(self, a):
        return np.abs(a)
    
    def sqrt(self, a):
        return np.sqrt(a)
    
    def random(self, size):
        return np.random.random_sample(size)
    
    def round(self, a, dec):
        return np.round(a, decimals=dec)
    
    def shape(self, a):
        return a.shape 
    
    def array_equal(self, a, b):
        return np.array_equal(a, b)
    
    def exp(self, a):
        return np.exp(a, dtype=self.dtype)
    
    def pi(self):
        return np.pi
    
    def arange(self, a):
        return np.arange(a)
    
    def copy_to_np_CPU(self, a):
        res = a
        if isinstance(a, np.ndarray):
            if a.size == 1:
                res = a.item()
        return res
    
    def is_complex(self, a):
        return np.iscomplexobj(a)
    
    def real_if_close(self, a, tol:float=1e-10):
        return np.real_if_close(a, tol=tol)


###########################################
############## Torch Backend ##############
###########################################

class torch_backend:
    def __init__(self, dtype: str='complex128', device: str='cpu'):
        if isinstance(dtype, str):
            if dtype == 'complex128':
                dtype = torch.complex128
            elif dtype == 'complex64':
                dtype = torch.complex64
            else:
                raise ValueError(f'Could not parse numpy dtype from "{dtype}"')
            
        if device == 'gpu' or device =='GPU' or device == 'cuda' or device == 'CUDA':
            self.device = 'cuda'
        elif device == 'cpu' or device == 'CPU':
            self.device = 'cpu'
        else:
            raise ValueError(f'Backend does not support "{device}" as device. Try: CPU or GPU')
        
        self.dtype  = dtype
        self.name = 'torch'

    def start_stop_time(self, t0=0):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return t1, t1-t0

    def asarray(self, a):
        return torch.tensor(a, device=self.device, dtype=self.dtype)
    
    def tensordot(self, a, b, axes, out=None):
        return torch.tensordot(a, b, axes, out=out)

    def transpose(self, a, perm=(1,0)):
        return torch.permute(a, perm)

    def conj(self, a):
        return torch.conj(a)
    
    def dagger(self,a):
        return torch.conj(a.T)

    def reshape(self, a, shape):
        return torch.reshape(a, shape)

    def svd(self, a, compute_uv=True):
        if compute_uv:
            return torch.linalg.svd(a, full_matrices=False)
        else:
            return torch.linalg.svdvals(a)
    
    def qr(self, a):
        return torch.linalg.qr(a)
    
    def lq(self, a):
        q, r = torch.linalg.qr(a.T)
        return r.T, q.T
    
    def norm(self, a):
        return torch.linalg.norm(a)

    def norm_sq(self, a):
        return torch.real(torch.sum(a * torch.conj(a)))

    def real(self, a):
        return torch.real(a)
    
    def dot(self, a, b):
        return torch.dot(a, b)
    
    def log(self, a):
        return torch.log(a)
    
    def eye(self, n, m=None, k=0):
        return torch.tensor(np.eye(n, M=m, k=k), device=self.device, dtype=self.dtype)
    
    def diag(self, a):
        return torch.diag(self.asarray(a))
    
    def copy(self, a):
        return torch.clone(a)
    
    def eig(self, a):
        return torch.linalg.eig(a)
    
    def argsort(self, a):
        return torch.argsort(a)
    
    def zeros(self, size):
        return torch.zeros(size, device=self.device, dtype=self.dtype)
    
    def ones(self, size, dtype=None):
        if dtype is None: dtype = self.dtype
        return torch.ones(size, device=self.device, dtype=dtype)
    
    def sum(self, a):
        return torch.sum(a)
    
    def flip(self, a, axis=[0]):
        return torch.flip(a, dims=axis)
    
    def trace(self, a):
        return torch.trace(a)
    
    def abs(self, a):
        return torch.abs(a)
    
    def sqrt(self, a):
        return torch.sqrt(a)
    
    def random(self, size):
        return torch.rand(size, device=self.device, dtype=self.dtype)
    
    def round(self, a, dec):
        return torch.round(a, decimals=dec)
    
    def shape(self, a):
        return a.size()
    
    def array_equal(self, a, b):
        return torch.equal(a, b)
    
    def exp(self, a):
        return torch.exp(a)
    
    def pi(self):
        return torch.pi
    
    def arange(self, a):
        return torch.arange(a)
    
    def copy_to_np_CPU(self, a):
        if type(a) is list:
            a_list = a
            is_list = True
        else:
            a_list = [a]
            is_list = False
        res_list = []
        for var in a_list:
            try:
                try:
                    res = (var.data).cpu().numpy()
                except:
                    res = (var.data).numpy()
            except:
                try:
                    res = np.asarray(var.cpu())
                except:
                    res = np.asarray(var)
            if res.size == 1:
                res = res.item()
            res_list.append(res)
        if is_list:
            res = res_list
        else:
            res = res_list[0]
        return res

    def is_complex(self, a):
        return torch.is_complex(a)
    
    def real_if_close(self, a, tol:float=1e-10):
        b = a.real+1j*self.zeros(self.shape(a))
        res_list = torch.isclose(a, b, rtol=tol, atol=tol)
        if torch.all(res_list == True):
            res = a.real
        else:
            res = a
        return res
