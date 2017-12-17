import numpy as np
# import scipy as sp
from scipy.integrate import ode
import time
import random
import pandas as pd

# Universal constants
kB = 3.16572e-6  # [Ha K^-1] # [m^2 kg s^-2 K^-1]

def mass(atom="Li"):
    if atom == "Li":
        return 7 *0.00054858 # electron mass units
    elif atom == "Cs":
        return 133 *0.00054858
m = 7 *0.00054858 # electron mass units  #* 1.66e-27  # [kg]

def C6(atom = "Li"):
    if atom == "Li":
        return 1394# C6 for Li+Li ai a.u.
    elif atom == "Cs":
        return 6891;
    else:
        return 0
def De(atom = "Li"):
    if atom == "Li":
        return 334*4.5563*10**(-6) # De for Li+Li in units of Ha
    elif atom == "Cs":
        return 279*4.5563*10**(-6)

def C12(atom = "Li"):
    return C6(atom)**2 / (4*De(atom)) # C12 coefficient given atom-atom depth De

def VLJ(r, atom = "Li"):
    '''Lennard-Jones potential for a pair of atoms separated by distance r.'''
    return C12(atom) / r ** 12 - C6(atom) / r ** 6

def Vtot(r1, r2, r3, atom = "Li"):
    '''Total potential energy for all three atoms'''
    r12 = np.linalg.norm(r1 - r2)
    r23 = np.linalg.norm(r2 - r3)
    r13 = np.linalg.norm(r1 - r3)
    return VLJ(r12, atom) + VLJ(r23, atom) + VLJ(r13, atom)


def dVdr(rA, rB, rC, atom = "Li"):
    '''form of derivative of V for particle A interacting with B and C'''
    rAB = rA - rB
    rAC = rA - rC
    rABnorm = np.linalg.norm(rAB)
    rACnorm = np.linalg.norm(rAC)
    if (rABnorm > 0 and rACnorm > 0):
        deriv_x = (12 * C12(atom) * rAB[0]) / rABnorm ** 14 - (6 * C6(atom) * rAB[0]) / rABnorm ** 8 + \
                  (12 * C12(atom) * rAC[0]) / rACnorm ** 14 - (6 * C6(atom) * rAC[0]) / rACnorm ** 8
        deriv_y = (12 * C12(atom) * rAB[1]) / rABnorm ** 14 - (6 * C6(atom) * rAB[1]) / rABnorm ** 8 + \
                  (12 * C12(atom) * rAC[1]) / rACnorm ** 14 - (6 * C6(atom) * rAC[1]) / rACnorm ** 8
        deriv_z = (12 * C12(atom) * rAB[2]) / rABnorm ** 14 - (6 * C6(atom) * rAB[2]) / rABnorm ** 8 + \
                  (12 * C12(atom) * rAC[2]) / rACnorm ** 14 - (6 * C6(atom) * rAC[2]) / rACnorm ** 8
    else:
        deriv_x = 1000 * np.ones(3)
        deriv_y = 1000 * np.ones(3)
        deriv_z = 1000 * np.ones(3)

    return np.array([deriv_x, deriv_y, deriv_z])


# rhs of ODE
def f_ODE(t, s, atom="Li"):
    r1 = s[0:3]
    r2 = s[3:6]
    r3 = s[6:9]
    v1 = s[9:12]
    v2 = s[12:15]
    v3 = s[15:18]

    dV1 = dVdr(r1, r2, r3, atom)
    dV2 = dVdr(r2, r1, r3, atom)
    dV3 = dVdr(r3, r1, r2, atom)

    output = np.array([v1, v2, v3, dV1, dV2, dV3]).flatten()
    return output


def run_collision(T, theta=np.pi / 2, b=0, t=(0, 10000), dt = 1, rinit1=150.0, d_molec = 8.78, max_step=0.1, atom="Li"):
    '''
    Runs colision simulation and returns scattering time, as well as vectors of atoms' positions.

    Parameters:
    ----------
    T : float
        The temperature of the incoming atom, setting the the collision energy
    theta : float
        The angle of the molecule (Defaults to Pi/2, so molecule is perpendicular to atom's initial trajectory)
    b : float
        The impact parameter of the collision (Defaults to zero)
    t : float tuple (t0, t1)
        The initial and final time of the intergation (Defaults to (0, 100))
    dt : float
        The time step of integration (Defaults to 0.01)
    rinit1: float
        The initial position of the atom (Defaults to 60.0)

    '''
    # initial time and end time
    t0, t1 = t
    # initial positions of atoms
    # distance between dimer's atoms
    #d_molec = 3#8.78
    r0 = np.array([[-rinit1 / 3, 0, 0],
                   [rinit1 / 6 + d_molec * np.cos(theta) / 2, -d_molec * np.sin(theta) / 2, 0],
                   [rinit1 / 6 - d_molec * np.cos(theta) / 2, d_molec * np.sin(theta) / 2, 0]])
    #r0 *= 5.29177*10 **(-11)

    # initial velocities of each atom
    vinit = -np.sqrt(kB * T / mass(atom)) / 10 # in nm/ns
    v0 = np.array([[-vinit, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    # group into vector  of ICs
    s0 = np.array([r0, v0])
    s0 = s0.flatten()

    # set up integrator
    r = ode(f_ODE).set_integrator('dop853', atol = 0.001)
    r.set_initial_value(s0, t0)

    # position vectors
    x1out = np.zeros(int(np.ceil(t1 / dt)))
    y1out = np.zeros(int(np.ceil(t1 / dt)))
    z1out = np.zeros(int(np.ceil(t1 / dt)))
    x2out = np.zeros(int(np.ceil(t1 / dt)))
    y2out = np.zeros(int(np.ceil(t1 / dt)))
    z2out = np.zeros(int(np.ceil(t1 / dt)))
    x3out = np.zeros(int(np.ceil(t1 / dt)))
    y3out = np.zeros(int(np.ceil(t1 / dt)))
    z3out = np.zeros(int(np.ceil(t1 / dt)))
    n = 0
    hyperrad = rinit1 / 2

    while hyperrad < rinit1 and r.t < t1:
        ds = r.integrate(r.t + dt)
        x1out[n] = ds[0] + vinit * r.t / 3
        y1out[n] = ds[1]
        z1out[n] = ds[2]
        x2out[n] = ds[3] + vinit * r.t / 3
        y2out[n] = ds[4]
        z2out[n] = ds[5]
        x3out[n] = ds[6] + vinit * r.t / 3
        y3out[n] = ds[7]
        z3out[n] = ds[8]
        hyperrad = np.sqrt(np.sum(ds[0:9] ** 2))
        n += 1

    x1out = x1out[0:n]
    y1out = y1out[0:n]
    z1out = z1out[0:n]
    x2out = x2out[0:n]
    y2out = y2out[0:n]
    z2out = z2out[0:n]
    x3out = x3out[0:n]
    y3out = y3out[0:n]
    z3out = z3out[0:n]

    # check which of the three atoms emerges independently
    r1_final = np.array([x1out[-1], y1out[-1], z1out[-1]])
    r2_final = np.array([x2out[-1], y2out[-1], z2out[-1]])
    r3_final = np.array([x3out[-1], y3out[-1], z3out[-1]])
    r12 = np.sqrt(np.sum((r1_final - r2_final) ** 2))
    r13 = np.sqrt(np.sum((r1_final - r3_final) ** 2))
    r23 = np.sqrt(np.sum((r3_final - r2_final) ** 2))

    # print("r12 %0.2f, r13 %0.2f, r23 %0.2f" %(r12, r13, r23))
    basin = 0  # the three atoms emerge separately
    if r12 < 11:
        basin = 1
    elif r13 < 11:
        basin = 2
    elif r23 < 11:
        basin = 3

    # Remember to convert time from atomic units (Hartree/h_bar) to ns
    return [r.t*2.41888*10**(-8), basin, x1out, y1out, z1out, x2out, y2out, z2out, x3out, y3out, z3out]


def delta_collision(delta, T, theta, atom="Li"):
    '''
    Runs two collision simulations with perturbation delta and returns whether or not the two trajectories
    end up in the same basin.

    Parameters:
    ----------
    delta : float
            perturbation of initial parameter (in our case just theta)
    T :     int
            temperature of collision
    theta : float
            angle orientation of dimer
    atom : string
            atom species considered
    Returns:
    ----------
    df_temp : pandas data frame
            contains all the information about the two trajectories
            (including whether they are stable under the perturbation delta)
    '''


    # Run collision with some theta_0 and store all the info
    collision_1 = run_collision(T=T, theta=theta, atom=atom)
    df_temp = pd.DataFrame([(0, atom, T, theta, delta, collision_1[0], collision_1[1],collision_1[2],collision_1[3],collision_1[4],
                            collision_1[5],collision_1[6],collision_1[7],collision_1[8],collision_1[9],collision_1[10])],
                           columns=['unstable', 'species', 'T', 'theta', 'delta', 'lifetime', 'basin',
                                    'x1out', 'y1out', 'z1out', 'x2out', 'y2out', 'z2out', 'x3out', 'y3out', 'z3out'])
    basin_1 = collision_1[1]

    # Ru collision with theta_0 + delta, compare the basin (so stbaility) and store the info
    collision_2 = run_collision(T, theta = theta + delta, atom=atom)
    basin_2 = collision_2[1]

    if basin_1 == basin_2:
        unstable = 0 # the analyzed trajectory is stable under perturbation delta
    else:
        unstable = 2 # the sum of the 'unstable column' ivided by the total number of runs will give the number of unstable trajectories

    df_temp = df_temp.append(pd.DataFrame([(unstable, atom, T, theta+delta, delta, collision_2[0], collision_2[1],collision_2[2],collision_2[3],collision_2[4],
                            collision_2[5],collision_2[6],collision_2[7],collision_2[8],collision_2[9],collision_2[10])],
                           columns=['unstable', 'species', 'T', 'theta', 'delta', 'lifetime', 'basin',
                                    'x1out', 'y1out', 'z1out', 'x2out', 'y2out', 'z2out', 'x3out', 'y3out', 'z3out']))

    return df_temp


def run_delta_collisions(N, delta, T, theta_range, atom= "Li"):
    '''
    Runs N sets of collision simulations that differ by perturbation delta.
    Returns (and saves to .csv file) data frame that contains all the relevant information about the considered trajectories,
    including whether they were stable under perturbation.

    Parameters
    ----------
    N : number of trajectories simulated
    delta :  perturbation of initial parameter (in our case just theta)
    T : temperature of collision
    theta_range : range of considered angles of the dimer
    atom : atom species

    '''
    # Pandas data frame that contains all the information about the considered trajectories
    pd_df = pd.DataFrame(columns=['unstable', 'species', 'T', 'theta', 'delta', 'lifetime', 'basin', 'x1out', 'y1out', 'z1out',
                                  'x2out', 'y2out', 'z2out', 'x3out', 'y3out', 'z3out'])

    for i in range(N):
        theta = random.uniform(theta_range[0], theta_range[1])
        delta_coll_result = delta_collision(delta, T, theta, atom)
        pd_df = pd_df.append(delta_coll_result)

    # data folder name
    data_name = (("data_%s") %(atom))
    # data file name
    t = int(time.time()) # use timestamp to differentiate different files (to second precision)
    # Also store temperature, number and delta information in the filename
    file_name = ("%s/out_%s_T%s_N%s_%.2E.csv" %(data_name, t, T, N, delta))

    pd_df.to_csv(file_name)

    return pd_df