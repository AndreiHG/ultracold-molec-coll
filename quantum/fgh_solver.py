import numpy as np
import matplotlib.pyplot as plt

def V_morse(x):
    D = 0.1744 # 4.7457 eV
    beta = 1.02764 # 1.94196e10 m-1
    x_e = 1.40201 # 0.74191e-10 m

    return D*(1-np.exp(-beta*(x-x_e)))**2

def V_ho(x):
    return 0.5*x**2

def fgh(V, x_grid, m):
    
    N = x_grid.size
    delta_x = x_grid[1] - x_grid[0]
    xmax = x_grid[-1]

    H = np.zeros((N,N))

    T_k = np.zeros(N)
    for n in range(0,N):
        T_k[n] = 2/m * (np.pi*(n-(N-1)/2.0)/(xmax))**2
    T_k = np.fft.fftshift(T_k)

    for n in range(0,N):

        phi_n = np.zeros(N)
        phi_n[n] = 1

        V_phi_n = np.zeros(N)
        V_phi_n[n] = V(x_grid[n])

        # fast fourier transform
        F_phi_n = np.fft.fft(phi_n) # into k-space
        TF_phi_n = np.multiply(T_k, F_phi_n)
        T_phi_n = np.fft.ifft(TF_phi_n) # not sure if this is necessary

        H[:,n] = T_phi_n + V_phi_n

    # diagonalize
    E, psi = np.linalg.eig(H)
    E = np.real(E)

    # sort energies in increasing order
    ind = np.argsort(E)
    E = E[ind]
    psi = psi[:,ind]

    print(E[0:20])
    plt.plot(x_grid,psi[:,5])
    plt.show()

if __name__ == '__main__':
    
    x_grid = np.linspace(0,10,1000)

    #plt.plot(x_grid,V_morse(x_grid)/0.1744)
    #plt.show()
    fgh(V_morse,x_grid,1822/2.0)