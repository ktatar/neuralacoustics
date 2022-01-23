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



    #--------------------------------------------------------------
    # set parameters

    # propagation params, explained in solver
    # potentially model can have different values across domain
    mu = torch.ones(b, h, w) * mu
    rho = torch.ones(b, h, w) * rho
    gamma = torch.ones(b, h, w) * gamma

    # initial condition
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    xi0 = torch.randn(b, h, w)
    # examples of other initial conditions
    #xi0[:,10:30,10:30] = noise[:,:,:]
    #xi0 = torch.zeros(b, h-2, w-2)
    #excitation_x = w//2
    #excitation_y = h//2
    #xi0[:, excitation_y, excitation_x] = 1

    # boundaries
    # if not specified, default boundary frame
    # yet, potentially model can have boundaries all over the domain, other than default boundary frame
    #bounds = torch.zeros(b, h, w)
    #bounds[:,h//2,:] = 1
    

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
    sol, sol_t = solver.run("cpu", dt, nsteps, b, w, h, mu, rho, gamma, xi0[:, 1:h-1, 1:w-1], torch.empty(0, 1), torch.empty(0, 1), disp, dispRate)

    return [sol, sol_t, xi0]


def getSolverInfo():
    if solver == 0:
        print(f'{model_name}: Cannot get description of solver! Model needs to be run at least once')
        return {'description':  ''}
    return solver.getInfo()
    