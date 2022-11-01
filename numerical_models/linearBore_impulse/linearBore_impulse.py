import torch
from pathlib import Path # to properly handle paths and folders on every os
from neuralacoustics.utils import openConfig
from neuralacoustics.utils import import_file

# to store values from load()
solver = 0 # where to load solver
modelName = ''
w = -1
h = -1
dt = -1
nsteps = -1
c = torch.empty((1,)) # declared as tensor to facilitate vector operations
rho = torch.empty((1,))
mu = torch.empty((1,))
tube_x = torch.empty((1,), dtype = torch.long) # declared as longs in order to allow indexing
tube_y = torch.empty((1,), dtype = torch.long)
tube_length = torch.empty((1,), dtype = torch.long)
tube_width =  torch.empty((1,), dtype = torch.long)
ex_mag = torch.empty((1,))

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
    _load(solver, prj_root, config_path)  # loads solver

    return

def load_test(config_path, prj_root):    
    # to prevent python from declaring new local variables with the same names
    # only needed when content of variables is modified
    global modelName 
    global w 
    global h 
    global dt 
    global nsteps 
    global c 
    global rho 
    global mu
    global tube_x
    global tube_y
    global tube_length
    global tube_width
    global ex_mag
    
    # get config file
    modelName = Path(__file__).stem #extracts modeName from this filename, stores it in global variable
    config = openConfig(config_path, modelName) 

    #--------------------------------------------------------------

    # read from config file
    # solver
    solver_path = config['solver'].get('solver')

    #simulation parameters (hardcoded for now)
    
    #parses all simulation parameters
    w = config['numerical_model_parameters'].getint('w') # width of grid
    h = config['numerical_model_parameters'].getint('h') # height of grid
    dt = 1.0/config['numerical_model_parameters'].getfloat('samplerate') #uses samplerate from ini file to calculate.
    nsteps = config['numerical_model_parameters'].getint('nsteps')#number of steps
    
    c[0] = config['numerical_model_parameters'].getfloat('c') # type of edge, 0 if clamped edge, 1 if free edge
    mu[0] = config['numerical_model_parameters'].getfloat('mu') # damping factor, positive and typically way below 1
    rho[0] = config['numerical_model_parameters'].getfloat('rho') # 'propagation' factor, positive and lte 0.5; formally defined as rho = [c*ds/dt)]^2, with c=speed of sound in medium, ds=size of each grid point [same on x and y], dt=1/samplerate
    
    tube_x[0] = int(eval(config['numerical_model_parameters'].get('tube_x'))) # tube params (position, length, width)
    tube_y[0] = int(eval(config['numerical_model_parameters'].get('tube_y'))) 
    tube_length[0] = int(eval(config['numerical_model_parameters'].get('tube_length') ))
    tube_width[0] = int(eval(config['numerical_model_parameters'].get('tube_width'))) 
    
    ex_mag[0] = config['numerical_model_parameters'].getfloat('ex_mag')#magnitude of excitation velocity [m/s]
    #--------------------------------------------------------------

    # load
    _load(solver_path, prj_root, config_path)  # loads solver
    
    return

def _load(solver_path, prj_root, config_path):
    global solver
    solver, temp_var = import_file(prj_root, config_path, solver_path)

    return

def run(dev, b, dt, nsteps, w, h, mu, rho, c, tube_x, tube_y, tube_length, tube_width, ex_mag, disp=False, dispRate=1, pause=0):
    #function will be called by generator, all params passed at runtime (does not use global variables)
    
    #--------------------------------------------------------------
    # initial condition
    
    # source direction input tensor, where zeros mean not excitation
    srcDir = torch.zeros([b, h, w, 4], device=dev) # last dim is left, down, right and top direction flag
    # excite -> if single slice, it's initial conditions only
    #VIC test full band impulse
    exciteV = torch.zeros([b, h, w, nsteps], device=dev)
    
    #tube walls, where zeros mean if not wall
    walls = torch.zeros([b, h, w], device=dev) # 1 is wall, 0 is not wall
    
    
    excitationW = 1
    for _b in range(b): 
        #always makes right-facing tube
        # top
        walls[_b, tube_y[_b]-1, tube_x[_b]:tube_x[_b]+tube_length[_b]] = 1
        # bottom
        walls[_b, tube_y[_b]+tube_width[_b], tube_x[_b]:tube_x[_b]+tube_length[_b]] = 1
            
        # define src directions
        srcDir[_b, tube_y[_b]:tube_y[_b]+tube_width[_b], tube_x[_b]:tube_x[_b]+excitationW, 0] = 0 # left
        srcDir[_b, tube_y[_b]:tube_y[_b]+tube_width[_b], tube_x[_b]:tube_x[_b]+excitationW, 1] = 0 # down
        srcDir[_b, tube_y[_b]:tube_y[_b]+tube_width[_b], tube_x[_b]:tube_x[_b]+excitationW, 2] = 1 # right
        srcDir[_b, tube_y[_b]:tube_y[_b]+tube_width[_b], tube_x[_b]:tube_x[_b]+excitationW, 3] = 0 # top
        
        #srcDir[:, tube_y[0]:tube_y[0]+tube_width[0], tube_x[0]:tube_x[0]+excitationW, direction[0]] = 1 
        
        # excitation
        exciteV[_b, tube_y[_b]:tube_y[_b]+tube_width[_b], tube_x[_b]:tube_x[_b]+excitationW, 0] = ex_mag[_b] # initial condition
    #--------------------------------------------------------------
    # run solver
    sol, sol_t = solver.run(dev, dt, nsteps, b, w, h, c, rho, mu, srcDir, exciteV, walls, disp=disp, dispRate=dispRate, pause=pause)
    return [sol, sol_t]

def run_test(dev, dispRate=1, pause=0):
    # set parameters
    _b = 1
    _disp = True
    _dispRate = 1/1
    _pause = 0

    #call run using those parameters+global variables, and return the result.
    test_sol, test_sol_t = run(dev, _b, dt, nsteps, w, h, mu, rho, c, tube_x, tube_y, tube_length, tube_width, ex_mag, disp =_disp, dispRate=_dispRate, pause=_pause)
    
    return [test_sol, test_sol_t]


def getSolverInfo():
    if solver == 0:
        print(f'{modelName}: Cannot get description of solver! Model needs to be run at least once')
        return {'description':  ''}
    return solver.getInfo()
    
