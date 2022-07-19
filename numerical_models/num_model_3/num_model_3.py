import torch
import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os
import random as rd
from neuralacoustics.utils import openConfig


# to store values from load()
solver = 0 # where to load solver
modelName = ''
w = -1
h = -1
mu = -1
rho = -1
gamma = -1
init_size_min = -1
init_size_max = -1
ex_x = -1
ex_y = -1
ex_size = -1


# places random impuse randomly in the matrix passed as an argument (xi0)
def add_random_impulse(b_size, xi_imp, w, h): 
    rand_pos_x = torch.randint(w, (b_size,1))
    rand_pos_y = torch.randint(h, (b_size,1))
    xi_imp[range(b_size), rand_pos_y[:,0], rand_pos_x[:,0]] = torch.randn(b_size) # places random impulse in location
    # tensor indexing with range() instead of ':' adapted from here: https://stackoverflow.com/questions/61096522/pytorch-tensor-advanced-indexing

def load(model_name, config_path, _w, _h):    
    # to prevent python from declaring new local variables with the same names
    # only needed when content of variables is modfied
    global solver 
    global modelName
    global w
    global h
    global mu
    global rho
    global gamma
    global init_size_min
    global init_size_max
    global ex_x
    global ex_y
    global ex_size

    modelName = model_name

     # get config file
    config = openConfig(config_path, modelName) 
    
    w = _w
    h = _h

    #--------------------------------------------------------------

    # read from config file
    # solver
    solver_dir = config['solver'].get('solver_dir')
    solver_name = config['solver'].get('solver_name')

    # simulation parameters
    mu = config['numerical_model_parameters'].getfloat('mu') # damping factor, positive and typically way below 1
    rho = config['numerical_model_parameters'].getfloat('rho') # 'propagation' factor, positive and lte 0.5; formally defined as rho = [c*ds/dt)]^2, with c=speed of sound in medium, ds=size of each grid point [same on x and y], dt=1/samplerate
    gamma = config['numerical_model_parameters'].getfloat('gamma') # type of edge, 0 if clamped edge, 1 if free edge
    
    ex_x = config['numerical_model_parameters'].getfloat('ex_x') # x value of excitation
    ex_y = config['numerical_model_parameters'].getfloat('ex_y') # y value of excitation (maybe these should be ints?)
    ex_size = config['numerical_model_parameters'].getfloat('ex_size') # amplitude value of excitation
   
    seed = config['numerical_model_parameters'].getfloat('seed') # pseudo random seed, for determinism

    #--------------------------------------------------------------

    # for determinism of initial conditions
    torch.manual_seed(seed)
    rd.seed(seed)
    

    #--------------------------------------------------------------

    # load solver
    # we want to load the package through potential subfolders
    solver_dir_folders = Path(solver_dir.replace('PRJ_ROOT', '.')).parts

    # create package structure by concatenating folders with '.'
    packages_struct = solver_dir_folders[0]
    for pkg in range(1,len(solver_dir_folders)):
        packages_struct += '.'+solver_dir_folders[pkg] 
    # load
    solver = __import__(packages_struct + '.' + solver_name, fromlist=['*']) # i.e., all.packages.in.solver.dir.solver_name

    return

def run_indv(dev, dt, nsteps, b=1, disp=False, dispRate=1, pause=0):

    # set parameters
    # propagation params, explained in solver
    # potentially model can have different values across domain
    _mu = torch.ones(1, h, w) * mu
    _rho = torch.ones(1, h, w) * rho
    _gamma = torch.ones(1, h, w) * gamma

    #--------------------------------------------------------------
    # initial condition
    excite = torch.zeros(1, h-2, w-2, nsteps) 
    # initial condition is first excitation (for individual run, only need one point)
    #can specify size of impulse or have it be random:
    excite[0, ex_x, ex_y, 0] = ex_size #places excitation (with a specified size) at specified point.
    #excite[0, ex_x, ex_y, 0] = torch.randn(1) #places excitation (with a random size) at specified point.

    #--------------------------------------------------------------
    # run solver
    sol, sol_t = solver.run(dev, dt, nsteps, 1, w, h, _mu, _rho, _gamma, excite, torch.empty(0, 1), disp, dispRate, pause)

    return [sol, sol_t]


def run_batch(dev, dt, nsteps, b, disp=False, dispRate=1, pause=0):

    # set parameters

    # propagation params, explained in solver
    # potentially model can have different values across domain
    _mu = torch.ones(b, h, w) * mu
    _rho = torch.ones(b, h, w) * rho
    _gamma = torch.ones(b, h, w) * gamma


    #--------------------------------------------------------------
    # initial condition
    xi0 = torch.zeros(b, h-2, w-2) # everywhere but bondary frame
    add_random_impulse(b, xi0, w-2, h-2) 

    excite = torch.zeros(b, h-2, w-2, nsteps)
    # initial condition is first excitation
    excite[..., 0] = xi0[...]
    

    #--------------------------------------------------------------
    # run solver
    sol, sol_t = solver.run(dev, dt, nsteps, b, w, h, _mu, _rho, _gamma, excite, torch.empty(0, 1), disp, dispRate, pause)

    return [sol, sol_t]


def getSolverInfo():
    if solver == 0:
        print(f'{modelName}: Cannot get description of solver! Model needs to be run at least once')
        return {'description':  ''}
    return solver.getInfo()
    
