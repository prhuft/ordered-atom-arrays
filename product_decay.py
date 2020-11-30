"""
Collective radiance simulations - Multi-excitation space

P. Huft

This script solves for the time evolution of a product state
in a grid of two level atoms and saves the solution as a
csv file.
"""

## imports
from numpy import *
import numpy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from time import time
from random import random
import csv
from scipy.special import binom
from datetime import datetime as dt

# local 
from physconsts import *
from simfuncs import *

################################################################################
## Functions and global definitions
################################################################################

## constants setup
lmbda = 7.8e-7
k = 2*pi/lmbda
gamma = 1

## function definitions
unit = lambda v: array(v)/sqrt(dot(array(v),array(v)))
egen = eye(3) # generic basis set
ecart = [array([1,0,-1])/sqrt(2), 
         -1j*array([1,0,1])/sqrt(2), 
         array([0,1,0])]

r = lambda i,j: array([0, 
                       mod(j,sqrt(atomnum)) - mod(i,sqrt(atomnum)),
                       floor(j/sqrt(atomnum)) - floor(i/sqrt(atomnum))])

def gdotp(r,p):
    """
    Dipole radiation measured at r
    
    Args:
        r: distance from dipole to measurement location
        p: vector representation of source dipole
    Returns:
        y: G acting on p, a complex vector
    """
    y = exp(1j*k*sqrt(dot(r,r)))*(cross(cross(unit(r),p),unit(r))
                                   + (1/(dot(r,r)*k**2)  
                                   - 1j/(k*sqrt(dot(r,r)))) \
                                   *(3*unit(r)*dot(unit(r),p) - p)) \
                                 /(4*pi*sqrt(dot(r,r)))
    return y

def prod_state(states, thresh):
    """
    Compute the product state for an aggregrate quantum system
    
    Args:
        states: list of equal-length arrays where each array stores
            the amplitudes describing the state of an individual system.
        thresh: the minimum acceptible probability of measuring a given
            basis vector in the product space. all basis vectors with
            amplitudes c where abs(c) < sqrt(thresh) will be omitted from
            the product space, giving an approximate basis for the state.
    Return:
        pstate:
            the approximate product state vector
    """
    pstate = states[0]
    for i in range(1, len(states)):
        pstate = array([p for p in kron(pstate, states[i]) if abs(p) > sqrt(thresh)])
    return pstate

class ArrayMode:
    
    def __init__(self, vector, name, color):
        self.vector = vector
        self.name = name
        self.color = color
        
    def __repr__(self):
        return f"ArrayMode({self.name})"

################################################################################
## Simulation setup and run
################################################################################

## basis setup
atomnum = 25
excitations = 2 # highest number of excitations
one_atoms = range(atomnum)
two_atoms = []
for i in range(atomnum):
    for j in range(i):
        two_atoms.append(array([i,j]))
atom_states = list(one_atoms) + two_atoms
basisnum = len(atom_states)

## square grid setup
gridname = "square"
center_idx = int(floor(sqrt(atomnum)/2)*(sqrt(atomnum)+1))

## sym. mode
fmode = ArrayMode(unit(concatenate((full(atomnum,1,complex),
                                   zeros(basisnum-atomnum)))), 'Symmetric', 'blue')

## product state
beta = 1/sqrt(atomnum)
alpha = sqrt(1-beta**2)
pvec = prod_state([array([alpha, beta],complex) for j in range(atomnum)],1e-4)[1:]
assert basisnum == len(pvec)

# sort. not always sensible choice, but valid for ordering into 1-,2- excitation kets
pvec.sort()
pvec = flip(pvec)
print(f"{len(pvec)} basis states in product state approximation")
pmode = ArrayMode(pvec, "Product", 'green')

## params
unitp = egen[0] # atom polarization
d = 0.75
tmin = 0 
tmax = 100
numsteps = 100
tsteps = linspace(tmin, tmax, numsteps)

## setup initial state
psi0 = pmode.vector
soln = empty(len(tsteps),float)

## build and diagonalize the hamiltonian
t0 = time()
print("constructing hamiltonian...")
hamiltonian = empty((basisnum, basisnum), complex)
for i in range(atomnum):
    for j in range(atomnum):
        if i!=j:
            u = lmbda*d*r(i,j)
            hamiltonian[i,j] = 6*pi*gamma*dot(unitp, gdotp(u, unitp))/k
        else:
            hamiltonian[i,j] = 1j*gamma
for i in range(atomnum, basisnum):
    for j in range(atomnum, basisnum):
        if i!=j:
            atomsi = atom_states[i]
            atomsj = atom_states[j]
            inter = intersect1d(atomsi,atomsj)
            if len(inter) == 1:
                pair = [atom for atom in concatenate((atomsi,atomsj)) \
                        if atom not in inter]
                u = lmbda*d*r(*pair)
            hamiltonian[i,j] = 6*pi*gamma*dot(unitp, gdotp(u, unitp))/k
        else:
            hamiltonian[i,j] = 1j*gamma
print(f"constructed hamiltonian in {int(floor((time()-t0)/60))} minutes")
            
assert hamiltonian.shape == (basisnum,basisnum)
    
t0 = time()
print("diagonalizing hamiltonian...")
evals,evecs = la.eig(hamiltonian)
print(f"diagonalized hamiltonian in {int(floor((time()-t0)/60))} minutes")

## set up and solve time evolution
dpsi = lambda t,state: 1j*dot(hamiltonian,state)

t0 = time()
print("solving Schrodinger equation... is the cat alive?")
soln = solve_ivp(dpsi,[tmin,tmax],psi0,t_eval=tsteps,vectorized=True)
print("done. time to check the cat.")
print(f"sim ran for {int(floor((time()-t0)/60))} minutes")


################################################################################
## Results analysis and output
################################################################################

# transform to vector solution
vecsoln = soln.y.T

## write soln to file
now = dt.now()
timestr = now.strftime("%Y%m%d_%H_%M_%S")
fname = f'soln_prod_decay_{int(sqrt(atomnum))}x{int(sqrt(atomnum))}_grid_{timestr}.csv'
soln_to_csv(fname, vecsoln, tsteps)


