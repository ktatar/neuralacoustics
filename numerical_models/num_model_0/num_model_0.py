import torch
import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os

solver = 0 # where to load module

def run(dev, dt, nsteps, b, w, h, model_name, config_path, disp=False, dispRate=1):
    global solver # to prevent python from declaring solver as new local variable when used in this function

    # get config file
    config = configparser.ConfigParser(allow_no_value=True)

    try:
        with open(config_path) as f:
            config.read_file(f)
    except IOError:
        print(f'{model_name}: Config file not found --- \'{config_path}\'')
        quit()


    # read from config file
    # solver
    solver_dir = config['solver'].get('solver_dir')
    solver_name = config['solver'].get('solver_name')

    # simulation parameters
    mu = config['numerical_model_parameters'].getfloat('mu') # damping factor, positive and typically way below 1; defined as ( 1+(eta*dt)/2 )^-1, with eta=viscosity of medium and dt=1/samplerate
    rho = config['numerical_model_parameters'].getfloat('rho') # 'propagation' factor, positive and lte 0.5; defined as rho = [v*ds/dt)]^2, with v=speed of wave in medium, ds=size of each grid point [same on x and y] and dt=1/samplerate
    gamma = config['numerical_model_parameters'].getfloat('gamma') # type of edge, 0 if clamped edge, 1 if free edge
    seed = config['numerical_model_parameters'].getfloat('seed') # pseudo random seed, for determinism


    #--------------------------------------------------------------
    # set parameters

    # propagation params, explained in solver
    # potentially model can have different values across domain
    mu = torch.ones(b, h, w) * mu
    rho = torch.ones(b, h, w) * rho
    gamma = torch.ones(b, h, w) * gamma

    # boundaries
    # if not specified, default boundary frame
    # yet, potentially model can have boundaries all over the domain, other than default boundary frame
    # like this:
    #bounds = torch.zeros(b, h, w)
    #bounds[:,h//2,:] = 1

    # initial condition
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    xi0 = torch.randn(b, h-2, w-2) # -2 to leave boundary frame alone

    # examples of other initial conditions
    #xi0 = torch.zeros(b, h-2, w-2)
    #excitation_x = (w-2)//2
    #excitation_y = (h-2)//2
    #xi0[:, excitation_y, excitation_x] = 1

    excite = torch.zeros(b, h-2, w-2, nsteps+1) # nsteps+1 is the total duration of simulation -> initial condition+requested steps
    # initial condition is first excitation
    excite[..., 0] = xi0[...]

    # example of continuous excitation
    # import math 
    # freq = 6000
    # ex = torch.FloatTensor([math.sin(2*math.pi*n*freq*dt) for n in range(nsteps+1)]).reshape(1,nsteps+1).repeat([b,1])
    # excite = torch.zeros(b, h-2, w-2, nsteps+1) # nsteps+1 is the total duration of simulation -> initial condition+requested steps
    # excite[:, 10, 10, :] = ex[...] # 


    

    #--------------------------------------------------------------

    # load solver
    # we want to load the package through potential subfolders
    solver_path_folders = Path(solver_dir.replace('PRJ_ROOT', '.')).parts

    # create package structure by concatenating folders with '.'
    packages_struct = solver_path_folders[0]
    for pkg in range(1,len(solver_path_folders)):
        packages_struct += '.'+solver_path_folders[pkg] 
    # load
    solver = __import__(packages_struct + '.' + solver_name, fromlist=['*']) # i.e., all.packages.in.solver.dir.solver_name

    
    #--------------------------------------------------------------
    # run solver
    sol, sol_t = solver.run(dev, dt, nsteps, b, w, h, mu, rho, gamma, excite, torch.empty(0, 1), disp, dispRate)

    return [sol, sol_t]


def getSolverInfo():
    if solver == 0:
        print(f'{model_name}: Cannot get description of solver! Model needs to be run at least once')
        return {'description':  ''}
    return solver.getInfo()
    