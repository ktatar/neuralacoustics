import sys
sys.path.append("..") # move a dir up

import torch
import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os
import random as rd

solver = 0 # where to load module


# places a submatrix of noise randomly in the matrix passed as an argument (p0)
def add_random_noise(b_size, pnoise, w, h, min_side, max_side): 
  rand_size = rd.randint(min_side, max_side) # generates a random size for the submatrix, between min_side and max_side inclusive
  rand_pos_x = rd.randint(0, w-2-rand_size) # generates a position in p0 to place the submatrix (makes sure not to cut off the submatrix)
  rand_pos_y = rd.randint(0, h-2-rand_size)
  pnoise[:, rand_pos_y:(rand_pos_y + rand_size), rand_pos_x:(rand_pos_x + rand_size)] = torch.randn(b_size, rand_size, rand_size) # places the submatrix in p0



def run(dev, dt, nsteps, b, w, h, model_path, model_name, disp=False, dispRate=1) :
    global solver # to prevent python from declaring solver as new local variable when used in this function

    # get config file
    config = configparser.ConfigParser(allow_no_value=True)
    config_path = model_path.joinpath(model_name+'.ini') # model_path/model_name.ini 

    try:
        config.read(config_path)
    except FileNotFoundError:
        print(model_name + ': Config File Not Found In {}'.format(config_path))
        sys.exit()


    # read from config file
    # solver
    solver_path = config['solver'].get('solver_path')
    solver_name = config['solver'].get('solver_name')

    # simulation parameters
    mu = config['parameters'].getfloat('mu') # damping factor, positive and typically way below 1
    rho = config['parameters'].getfloat('rho') # 'propagation' factor, positive and lte 0.5; formally defined as rho = [c*ds/dt)]^2, with c=speed of sound in medium, ds=size of each grid point [same on x and y], dt=1/samplerate
    gamma = config['parameters'].getfloat('gamma') # type of edge, 0 if clamped edge, 1 if free edge
    # initial conditions---can be code
    init_size_min = eval( str( config['parameters'].get('init_size_min') ) )   # sub matrix min size [side] 
    init_size_max = eval( str( config['parameters'].get('init_size_max' ) ) )   # sub matrix min size [side] 

    s = min(w,h)-2 # remove two cells to accomodate boundary frame
    if init_size_max > s :
        init_size_max = s
        print(model_name + ': Requested init_size_max exceeds domain smaller dimension and will be clipped to {} cells'.format(init_size_max))

    #--------------------------------------------------------------
    # set parameters

    # propagation params, explained in solver
    # potentially model can have different values across domain
    mu = torch.ones(b, h, w) * mu
    rho = torch.ones(b, h, w) * rho
    gamma = torch.ones(b, h, w) * gamma

    # initial condition
    p0 = torch.zeros(b, h, w) # everywhere but bondary frame
    add_random_noise(b, p0[:, 1:h-1, 1:w-1], w, h, init_size_min, init_size_max) 
    #--------------------------------------------------------------

    # load solver
    # we expect a path starting with '../../' etc, to get to project root dir
    solver_path = solver_path.replace('../', '')
    solver_path = solver_path.replace('/', '.') # this is for potential subfolders
    solver = __import__(solver_path + '.' + solver_name, fromlist=['*']) # models_root.model_name.model_name is model script

    
    #--------------------------------------------------------------
    # run solver
    sol, sol_t = solver.run("cpu", dt, nsteps, b, w, h, mu, rho, gamma, p0[:, 1:h-1, 1:w-1], torch.empty(0, 1), torch.empty(0, 1), disp, dispRate)

    return [sol, sol_t, p0]


def getSolverDescription() :
    if solver == 0 :
        print(model_name + ': Cannot get description of solver! Model needs to be run at least once')
        return ''
    return solver.getDescription()
    