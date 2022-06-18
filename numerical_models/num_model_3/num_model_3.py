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


# places random impuse randomly in the matrix passed as an argument (xi0)
def add_random_impulse(b_size, xi_imp, w, h): 
  rand_pos_x = rd.randint(0, w-2) # generates a position in xi0 to place the impulse
  rand_pos_y = rd.randint(0, h-2)
  xi_imp[:, rand_pos_y, rand_pos_x] = torch.randn(b_size) # places random impulse in location


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


def run(dev, dt, nsteps, b, disp=False, dispRate=1, pause=0):

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
    