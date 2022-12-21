import torch
from pathlib import Path # to properly handle paths and folders on every os
from neuralacoustics.utils import openConfig
from neuralacoustics.utils import import_fromScript

N = -1
B = -1
w = -1
h = -1
mu = torch.empty((1,))
rho = torch.empty((1,))
gamma = torch.empty((1,))
nsteps = -1
dryrun = -1
dt = -1
pause_sec = -1
model = 0
generator_name = ''
input_grid = []
current_input = 0

def load(config_path, ch, prj_root, pause):
    # in same style as load_test, this function writes the config variables to global variables.
    global N 
    global B 
    global w 
    global h
    global mu
    global rho
    global gamma
    global nsteps 
    global dt
    global dryrun 
    global pause_sec
    global model
    global generator_name
    global input_grid

    # open config file
    generator_name = Path(__file__).stem
    config = openConfig(config_path, generator_name)
    
    #------------------------------------------------------------------------------------------------------------------------------------
    #parameters

    # model path
    model_path = config['dataset_generator_parameters'].get('numerical_model')  # path of num_model

    # model config file
    model_config_path = config['dataset_generator_parameters'].get('numerical_model_config')

    # dataset size
    B = config['dataset_generator_parameters'].getint('B') # batch size
    
    # seconds to pause between datapoints during visualization
    pause_sec = pause
    # only used in dry run and it will be ignored in solver if <= 0
    
    # domain size
    w = config['numerical_model_parameters'].getint('w') # domain width [cells]
    h = config['numerical_model_parameters'].getint('h') # domain height[cells]

    mu[0] = config['numerical_model_parameters'].getfloat('mu') 
    rho[0] = config['numerical_model_parameters'].getfloat('rho') 
    gamma[0] = config['numerical_model_parameters'].getfloat('gamma')
    
    # time parameters
    nsteps = config['numerical_model_parameters'].getint('nsteps') # = T_in+T_out, e.g., Tin = 10, T = 10 -> input [0,Tin), output [10, Tin+T)
    dt = 1.0 / config['numerical_model_parameters'].getint('samplerate') # seconds (1/Hz), probably no need to ever modify this...
    
    #get grid paramters as strings
    grid_bin = config['dataset_generator_parameters'].get('grid_bin')
    grid_mag = config['dataset_generator_parameters'].get('grid_mag')
    grid_phase = config['dataset_generator_parameters'].get('grid_phase')
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    
    N = 1
    
    #takes string, checks that the range matches the arguments:string, min, max (min, max both inclusive)
    #then returns the torch.linspace tensor created from that range
    linspace_bin = strToLinspace(grid_bin, 0, (w-2)//2, 'grid_bin')
    linspace_mag = strToLinspace(grid_mag, -1, 1, 'grid_mag')
    linspace_phase = strToLinspace(grid_phase, 0, 0.99, 'grid_phase')
    
    N_input_grid = N
    
    #check if we need to pad a batch with extra zero entries
    if N % B > 0:
        extra_zeros = B - (N % B)
        N_input_grid += extra_zeros
        print(f'{N} entries will be generated, with {extra_zeros} extra zeros, {N_input_grid} entries total.')
    else:
        print(f'{N_input_grid} entries will be generated.')
    
    #prepare input grid
    input_grid = torch.zeros(N_input_grid, 3) # first dimension is index of input, then spatial freq/magnitude/phase
    
    num_of_bins = len(linspace_bin)
    num_of_mags = len(linspace_mag)
    num_of_phases = len(linspace_phase)
    
    input_grid[:N, 0] = linspace_bin.repeat_interleave(num_of_mags* num_of_phases)  # uses repeat and repeat_interleave to generate all possible permutations
    input_grid[:N, 1] = linspace_mag.repeat_interleave(num_of_phases).repeat(num_of_bins)
    input_grid[:N, 2] = linspace_phase.repeat(num_of_bins* num_of_mags)

    #------------------------------------------------------------------------------------------------------------------------------------
    # imports + loads model

    model_function_list = ['load, run']  # specify which functions to load.
    model, model_config_path = import_fromScript(prj_root, config_path, model_path, model_config_path, function_list=model_function_list)

    model.load(model_config_path, prj_root)  # loads solver for model

    #---------------------------------------------------------------------------
    #compute meta data, e.g., duration, actual size...
    
    num_of_batches = N_input_grid//B
    
    # num of chunks must be lower than total number of batches
    if ch > num_of_batches:
      ch = num_of_batches//2 # otherwise, a chunk every other batch
    if ch == 0: # always at least one chunk!
      ch = 1

    return num_of_batches, ch, N, B, h, w, nsteps, dt, model_config_path

def generate_datasetBatch(dev, dryrun):
    
    if dryrun == 0:
        ex_input_freq, ex_input_mag, ex_input_phase = generateFromInputGrid(B)
        full_excitation, sol = model.run(dev, B, dt, nsteps, w, h, mu, rho, gamma, ex_input_freq, ex_input_mag, ex_input_phase)
        
    else:
        ex_input_freq, ex_input_mag, ex_input_phase = generateFromInputGrid(1) #create rand tensors for excitation and medium
        full_excitation, sol  = model.run(dev, 1, dt, nsteps, w, h, mu, rho, gamma, ex_input_freq, ex_input_mag, ex_input_phase, disp =True, dispRate = 1/1, pause = pause_sec) #run with B = 1
    
    return full_excitation, sol


def generateFromInputGrid(_B):
    global current_input
    
    input_freq = input_grid[current_input:current_input + _B, 0].long() #long to allow indexing
    magnitude = input_grid[current_input:current_input + _B, 1]
    phase = input_grid[current_input:current_input + _B, 2]
    
    current_input += _B
    
    return input_freq, magnitude, phase


def getSolverInfoFromModel():
    return model.getSolverInfo()


def strToLinspace(grid_params_string, min_bound, max_bound, name_string):
    global N
    
    grid_params = eval(grid_params_string) # in form [min, max, num of steps] (min and max are inclusive + num of steps includes them both)
    
    #check if values are out of bounds:
    if grid_params[0] < min_bound :
        grid_params[0] = min_bound
        print(f'{generator_name}: Requested minimum value for {name_string} below intended minimum, will be clipped to {min_bound:.2f}')
    if grid_params[1] > max_bound :
        grid_params[1] = max_bound
        print(f'{generator_name}: Requested maximum value for {name_string} above intended maximum, will be clipped to {max_bound:.2f}')
    
    N *= grid_params[2] #increase N
    lin_tensor = torch.linspace(*grid_params)
    return lin_tensor
