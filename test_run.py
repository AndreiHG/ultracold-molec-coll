import numpy as np
import multiprocessing
import molecCollPackage as molec
import sys

theta_range = np.array([0, np.pi/2])  #0 to Pi/2 covers all possibilities (assuming b (impact parameter) = 0 or homonuclear dimer)
cores = 8
delta_range = np.logspace(-12, -1, cores)

# Not 100% sure we need to declare the temperature and number of runs as global variables
global T
T = int(sys.argv[1])
global N
N = int(sys.argv[2])

def par_func(i):

    molec.run_delta_collisions(N, delta_range[i], T, theta_range, atom = "Li")
    return 0

if __name__=='__main__':
    pool = multiprocessing.Pool(processes=cores)
    pool.map(par_func, range(cores))
    pool.close()
    pool.join()

    # Only get here once all parallel processes have finished
