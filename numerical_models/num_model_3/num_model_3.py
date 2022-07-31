import torch
import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os
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
ex_amp = -1
dt = -1

def load():    
    global modelName 
    modelName = 'num_model_3' #sets model name

    #does not read/assign any parameters, called by generator which will pass a new set of parameters in every run call.
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

def load_test(config_path):    
    # to prevent python from declaring new local variables with the same names
    # only needed when content of variables is modified
    global solver 
    global w
    global h
    global mu
    global rho
    global gamma
    global init_size_min
    global init_size_max
    global ex_x
    global ex_y
    global ex_amp
    global dt

    load() #loads solver, which sets modelName to num_model_3
    
    # get config file
    config = openConfig(config_path, modelName) 

    #--------------------------------------------------------------

    # read from config file
    # solver
    solver_dir = config['solver'].get('solver_dir')
    solver_name = config['solver'].get('solver_name')

    # simulation parameters
    mu = config['numerical_model_parameters'].getfloat('mu') # damping factor, positive and typically way below 1
    rho = config['numerical_model_parameters'].getfloat('rho') # 'propagation' factor, positive and lte 0.5; formally defined as rho = [c*ds/dt)]^2, with c=speed of sound in medium, ds=size of each grid point [same on x and y], dt=1/samplerate
    gamma = config['numerical_model_parameters'].getfloat('gamma') # type of edge, 0 if clamped edge, 1 if free edge
    
    w = config['numerical_model_parameters'].getint('w') # width of grid
    h = config['numerical_model_parameters'].getint('h') # height of grid
    dt = 1.0/config['numerical_model_parameters'].getfloat('samplerate') #uses samplerate from ini file to calculate.
    
    ex_x = config['numerical_model_parameters'].getint('ex_x') # x value of excitation
    ex_y = config['numerical_model_parameters'].getint('ex_y') # y value of excitation
    ex_amp = config['numerical_model_parameters'].getfloat('ex_amp') # amplitude value of excitation
    
    #--------------------------------------------------------------
    
    return

def run(dev, b, samplerate, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, ex_amp, disp=False, dispRate=1, pause=0):
    #function will be called by generator, all params passed at runtime (does not use global variables)
    #The arguments mu, rho, gamma, ex_x, ex_y, ex_amp are all arrays with b elements.

    # set parameters

    # propagation params, explained in solver
    # potentially model can have different values across domain (for now, model params only differ by batch, not across domain)
    _mu = torch.ones(b, h, w) 
    _rho = torch.ones(b, h, w)
    _gamma = torch.ones(b, h, w)
    
    # element-wise multiplication of tensor slice and vector
    # adapted from here: https://discuss.pytorch.org/t/element-wise-multiplication-of-a-vector-and-a-matrix/56946
    _mu = mu.view(-1, 1, 1).expand_as(_mu) * _mu
    _rho = rho.view(-1, 1, 1).expand_as(_rho) * _rho
    _gamma = gamma.view(-1, 1, 1).expand_as(_gamma) * _gamma
    
    #--------------------------------------------------------------
    # initial condition
    excite = torch.zeros(b, h-2, w-2, nsteps)
    excite[range(b), ex_y[range(b)], ex_x[range(b)], 0] = ex_amp[range(b)]

    #--------------------------------------------------------------
    # run solver
    sol, sol_t = solver.run(dev, dt, nsteps, b, w, h, _mu, _rho, _gamma, excite, torch.empty(0, 1), disp, dispRate, pause)

    return [sol, sol_t]

def run_test(dev, dispRate=1, pause=0):
    # set parameters
    _b = 1
    _disp = True
    
    #call run using those parameters, and return the result.
    test_sol, test_sol_t = run(dev, _b, samplerate, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, ex_amp, _disp=False)
    
    return [test_sol, test_sol_t]


def getSolverInfo():
    if solver == 0:
        print(f'{modelName}: Cannot get description of solver! Model needs to be run at least once')
        return {'description':  ''}
    return solver.getInfo()
    
