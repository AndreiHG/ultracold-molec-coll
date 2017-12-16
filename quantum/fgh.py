import numpy as np
import scipy as sp
from scipy import optimize
import constants as c

class problem:

    def __init__(self,V,m,units='SI'):
        self.V = V
        self.m = m
        self.units = units

class grid:

    def __init__(self,q_min,q_max,N,cmap,dcmap):
        self.setup_grid(q_min,q_max,N,cmap,dcmap)

    def setup_grid(self,q_min,q_max,N,cmap,dcmap):
        mapping = lambda Q: q_min + cmap(Q)
        Q_max = sp.optimize.root(lambda Q: mapping(Q)-q_max,1.0).x
        Q_min = 0
        Q = np.linspace(Q_min,Q_max,N)
        q = mapping(Q)
        J = dcmap(Q)
        J_inv = J**(-1)
        
        self.N = N
        self.q_min = q_min
        self.q_max = q_max
        self.q = q
        self.Q_max = Q_max
        self.Q_min = Q_min
        self.Q = Q
        self.J = J
        self.J_inv = J_inv
        self.cmap = cmap
        self.dcmap = dcmap 

class solution:
    def __init__(self,E,psi,grid,prob):
        self.E = E
        self.psi = psi
        self.x = grid.q
        self.grid = grid
        self.prob = prob

def normalize(psi, grid):
    N = grid.N
    psi_norm = np.zeros(psi.shape)
    for i in range(0,N):
        p = psi[:,i]
        t = np.multiply(p, p)
        t = np.multiply(t, grid.J)
        psi_norm[:,i] = p / np.sqrt(t)
    return psi_norm    

def solve(prob, grid):
    
    N = grid.N
    Q_max_min = grid.Q_max - grid.Q_min

    H = np.zeros((N,N))

    T_k = np.zeros(N)
    for n in range(0,N):
        T_k[n] = n-(N-1)/2.0
    T_k = np.fft.fftshift(T_k)

    for n in range(0,N):

        V_phi_n = np.zeros(N)
        V_phi_n[n] = prob.V(grid.q[n])

        phi_n = np.zeros(N)
        phi_n[n] = 1

        # fast fourier transform
        F_phi_n = np.fft.fft(phi_n) # into k-space     
        dphi_dq_n = np.multiply(grid.J_inv, np.fft.ifft(np.multiply(T_k, F_phi_n)))
        F_dphi_dq_n = np.fft.fft(dphi_dq_n)
        dphi_dq2_n = np.multiply(grid.J_inv, np.fft.ifft(np.multiply(T_k, F_dphi_dq_n)))
        T_phi_n = dphi_dq2_n * 2/prob.m * (np.pi/(Q_max_min))**2
        if prob.units == 'SI':
            T_phi_n = T_phi_n * c.hbar**2

        H[:,n] = np.real(T_phi_n + V_phi_n)

    # diagonalize
    E, psi = np.linalg.eig(H)
    E = np.real(E)

    # sort energies in increasing order
    ind = np.argsort(E)
    E = E[ind]
    psi = psi[:,ind]

    # normalize
    psi_norm = normalize(psi, grid)
    
    return solution(E,psi,grid,prob)



