import torch
import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os
from neuralacoustics.utils import openConfig


# to store values from load()
solver = 0 # where to load solver
modelName = ''
w = -1
h = -1
dev = ''
dt = -1
nsteps = -1
c = -1
rho = -1
mu = -1
pmls = -1
pmlAttn = -1
excitationX = -1
excitationY = -1
excitationW = -1
excitationH = -1
tubeLength = -1

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
    global dev
    global dt 
    global nsteps 
    global c 
    global rho 
    global mu 
    global pmls 
    global pmlAttn 
    global excitationX
    global excitationY 
    global excitationW 
    global excitationH 
    global tubeLength 
    
    # get config file
    modelName = Path(__file__).stem #extracts modeName from this filename, stores it in global variable
    config = openConfig(config_path, modelName) 

    #--------------------------------------------------------------

    # read from config file
    # solver
    solver_path = config['solver'].get('solver')

    #simulation parameters (hardcoded for now)
    
    #---------------------------------------------------------------------
    #VIC these are the variables that will be passed by the model to the solver as input arguments

    dt = 1/44100.0
    nsteps = 200
    b = 5
    w = 64
    h = 64

    # these are not matrices cos otherwise we would not know what to put in pmls...
    # Sound speed in air [m/s] 
    c = 350  
     # Air density  [kg/m^3]
    rho = 1.14
    # wall admittance
    mu = 0.03
    
    # Number of PML layers
    pmls = 6      # default value is 6
    # max pml attenuation 
    pmlAttn = 0.5 # default value is 0.5

    #------------------------------------------
    #VIC here we initialize some of the variables that will be passed as arguments
    # we create a tube and we push a puff of air into its left end
    # # all this stuff should be moved to the model

    excitationX = w//4
    excitationY = h//2
    excitationW = 1
    excitationH = h//10
    tubeLength = w//3

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

def run(dev, b, dt, nsteps, w, h, mu, rho, c, excitationX, excitationY, excitationW, excitationH, tubeLength, pmls, pmlAttn, disp=False, dispRate=1, pause=0):
    #function will be called by generator, all params passed at runtime (does not use global variables)
    
    #--------------------------------------------------------------
    # initial condition
    
    # source direction input tensor, where zeros mean not excitation
    srcDir = torch.zeros([b, h, w, 4], device=dev) # last dim is left, down, right and top direction flag
    # excite -> if single slice, it's initial conditions only
    #VIC test full band impulse
    exciteV = torch.zeros([b, h, w, nsteps], device=dev)
    # walls, where zeros mean if not wall
    walls = torch.zeros([b, h, w], device=dev) # 1 is wall, 0 is not wall
    
    # tube wall
    # top
    walls[:, excitationY-1, excitationX:excitationX+tubeLength] = 1
    # bottom
    walls[:, excitationY+excitationH, excitationX:excitationX+tubeLength] = 1

    # define src directions
    srcDir[:, excitationY:excitationY+excitationH, excitationX:excitationX+excitationW, 0] = 0 # left
    srcDir[:, excitationY:excitationY+excitationH, excitationX:excitationX+excitationW, 1] = 0 # down
    srcDir[:, excitationY:excitationY+excitationH, excitationX:excitationX+excitationW, 2] = 1 # right
    srcDir[:, excitationY:excitationY+excitationH, excitationX:excitationX+excitationW, 3] = 0 # top

    # excitation
    exciteV[:, excitationY:excitationY+excitationH, excitationX:excitationX+excitationW, 0] = 0.01 # initial condition
    
    #--------------------------------------------------------------
    # run solver
    sol, sol_t = solver.run(dev, dt, nsteps, b, w, h, c, rho, mu, srcDir, exciteV, walls, pmls, pmlAttn , disp, dispRate, pause)
    return [sol, sol_t]

def run_test(dev, dispRate=1, pause=0):
    # set parameters
    _b = 1
    _disp = True
    _dispRate = 1/1
    _pause = 0

    #call run using those parameters+global variables, and return the result.
    test_sol, test_sol_t = run(dev, _b, dt, nsteps, w, h, mu, rho, c, excitationX, excitationY, excitationW, excitationH, tubeLength, pmls, pmlAttn, _disp, _dispRate, _pause)
    
    return [test_sol, test_sol_t]


def getSolverInfo():
    if solver == 0:
        print(f'{modelName}: Cannot get description of solver! Model needs to be run at least once')
        return {'description':  ''}
    return solver.getInfo()
    


