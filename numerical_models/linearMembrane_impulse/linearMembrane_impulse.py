import torch
import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os
from neuralacoustics.utils import openConfig


# to store values from load()
solver = 0 # where to load solver
modelName = ''
w = -1
h = -1
mu = torch.empty((1,)) # declared as tensor to facilitate vector operations
rho = torch.empty((1,))
gamma = torch.empty((1,))
init_size_min = -1
init_size_max = -1
ex_x = torch.empty((1,), dtype = torch.long) # declared as longs in order to allow indexing
ex_y = torch.empty((1,), dtype = torch.long) 
ex_amp = torch.empty((1,))
dt = -1
nsteps = -1


def load(config_path, prj_root):    
    global modelName

    # get config file
    modelName = Path(__file__).stem #extracts modeName from this filename, stores it in global variable
    config = openConfig(config_path, modelName) 

    #--------------------------------------------------------------

    # read from config file
    solver = config['solver'].get('solver')#solver path
    
    # other params will be passed to run() at run-time

    #--------------------------------------------------------------

    # load
    _load(solver, prj_root) #loads solver

    return

def load_test(config_path, prj_root):    
    # to prevent python from declaring new local variables with the same names
    # only needed when content of variables is modified
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
    global ex_amp
    global dt
    global nsteps
    
    # get config file
    modelName = Path(__file__).stem #extracts modeName from this filename, stores it in global variable
    config = openConfig(config_path, modelName) 

    #--------------------------------------------------------------

    # read from config file
    # solver
    solver_path = config['solver'].get('solver')

    #parses all simulation parameters
    mu[0] = config['numerical_model_parameters'].getfloat('mu') # damping factor, positive and typically way below 1
    rho[0] = config['numerical_model_parameters'].getfloat('rho') # 'propagation' factor, positive and lte 0.5; formally defined as rho = [c*ds/dt)]^2, with c=speed of sound in medium, ds=size of each grid point [same on x and y], dt=1/samplerate
    gamma[0] = config['numerical_model_parameters'].getfloat('gamma') # type of edge, 0 if clamped edge, 1 if free edge
    
    w = config['numerical_model_parameters'].getint('w') # width of grid
    h = config['numerical_model_parameters'].getint('h') # height of grid
    dt = 1.0/config['numerical_model_parameters'].getfloat('samplerate') #uses samplerate from ini file to calculate.
    nsteps = config['numerical_model_parameters'].getint('nsteps')#number of steps
    
    ex_x[0] = config['numerical_model_parameters'].getint('ex_x') # x value of excitation
    ex_y[0] = config['numerical_model_parameters'].getint('ex_y') # y value of excitation
    ex_amp[0] = config['numerical_model_parameters'].getfloat('ex_amp') # amplitude value of excitation
    
    #--------------------------------------------------------------

    # load
    _load(solver_path, prj_root) #loads solver
    
    return

def _load(solver_path, prj_root):    
    global solver
    
    solver_name = Path(solver_path).parts[-1]
    solver_path = solver_path + '/' + solver_name 
    
    #--------------------------------------------------------------
    # load solver
    # we want to load the package through potential subfolders
    solver_dir_folders = Path(solver_path.replace('PRJ_ROOT', prj_root)).parts # create full path [with no file extension] and get folders and file name
    
    # create package structure by concatenating folders with '.'
    packages_struct = '.'.join(solver_dir_folders)[:] # append all parts
    # load
    solver = __import__(packages_struct, fromlist=['*']) # i.e., all.packages.in.solver.dir.solver_name

    return    

def run(dev, b, dt, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, ex_amp, disp=False, dispRate=1, pause=0):
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
    # tensor indexing with range() instead of ':' adapted from here: https://stackoverflow.com/questions/61096522/pytorch-tensor-advanced-indexing

    #--------------------------------------------------------------
    # run solver
    sol, sol_t = solver.run(dev, dt, nsteps, b, w, h, _mu, _rho, _gamma, excite, torch.empty(0, 1), disp, dispRate, pause)

    return [sol, sol_t]

def run_test(dev, dispRate=1, pause=0):
    # set parameters
    _b = 1
    _disp = True
    
    #call run using those parameters+global variables, and return the result.
    test_sol, test_sol_t = run(dev, _b, dt, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, ex_amp, _disp, dispRate, pause)
    
    return [test_sol, test_sol_t]


def getSolverInfo():
    if solver == 0:
        print(f'{modelName}: Cannot get description of solver! Model needs to be run at least once')
        return {'description':  ''}
    return solver.getInfo()
    
