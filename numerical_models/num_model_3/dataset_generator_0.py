import torch
import configparser # to save config in new ini file
import scipy.io # to save dataset
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time
from neuralacoustics.utils import openConfig

numerical_model = ''
numerical_model_dir = ''
solver_dir = ''
solver_name = ''
N = -1
B = -1
w = -1
h = -1
seed = -1
nsteps = -1
dt = -1
pause_sec = -1
model = 0

def load(config_path, ch):
    # in same style as load_test, this function writes the config variables to global variables.
    global numerical_model 
    global numerical_model_dir 
    global solver_dir
    global solver_name 
    global N 
    global B 
    global w 
    global h
    global nsteps 
    global dt
    global dryrun 
    global pause_sec
    global model
    
    #open config file
    generator_name = Path(__file__).stem
    config = openConfig(config_path, generator_name)
    
    #------------------------------------------------------------------------------------------------------------------------------------
    #parameters
    
    # dataset size
    N = config['dataset_generation'].getint('N') # num of dataset points
    B = config['dataset_generation'].getint('B') # batch size
    
    # domain size
    w = config['dataset_generation'].getint('w') # domain width [cells]
    h = config['dataset_generation'].getint('h') # domain height[cells]
    
    # time parameters
    nsteps = config['dataset_generation'].getint('nsteps') # = T_in+T_out, e.g., Tin = 10, T = 10 -> input [0,Tin), output [10, Tin+T)
    dt = 1.0 / config['dataset_generation'].getint('samplerate') # seconds (1/Hz), probably no need to ever modify this...
    
    #for quick visualization
    dryrun = config['dataset_generation'].getint('dryrun') # visualize a single simulation run or save full dataset
    # seconds to pause between datapoints during visualization
    pause = config['dataset_generation'].getfloat('pause_sec')
    # only used in dry run and it will be ignored in solver if <= 0
    
    #seed
    seed = config['dataset_generation'].getint('seed') #for determinism.
    
    #solver parameters (used for model.load)
    solver_dir = config['solver'].get('solver_dir')
    solver_name = config['solver'].get('solver_name')
    
    #model
    numerical_model = config['dataset_generation'].get('numerical_model') #name of num_model
    numerical_model_dir = config['dataset_generation'].get('numerical_model_dir') #directory of num_model
    
    #------------------------------------------------------------------------------------------------------------------------------------
    #loads model
    
    model_path_folders = Path(numerical_model_dir.replace('PRJ_ROOT', '.')).joinpath(numerical_model).parts
    # ^ also add folder with same name as model (so we have ./{numerical_model_dir}/{numerical_model}/)

    # create package structure by concatenating folders with '.'
    packages_struct = model_path_folders[0]
    for pkg in range(1,len(model_path_folders)):
        packages_struct += '.'+model_path_folders[pkg] 
    
    # load 
    model = __import__(packages_struct + '.' + numerical_model, fromlist=['*']) # model.path.numerical_model is model script [i.e., package]
    model.load(solver_dir, solver_name) #loads solver for model
    
    #-----------------------------------------------------------------------------------------------------------------------------------
    torch.use_deterministic_algorithms(True) #enables determinism.
    torch.backends.cudnn.deterministic = True 
    torch.manual_seed(seed) #sets seed
    
    #----------------------------------------------------------------------------
    #compute meta data, e.g., duration, actual size...

    num_of_batches = N//B
    
    # num of chunks must be lower than total number of batches
    if ch > num_of_batches:
      ch = num_of_batches//2 # otherwise, a chunk every other batch
    if ch == 0: # always at least one chunk!
      ch = 1
        
    if dryrun:
        num_of_batches = -1
        
    rem = 0 # is there any remainder?
    
    return num_of_batches, ch, rem

def generate_datasetBatch(dev):
    if dryrun:
        ex_x, ex_y, ex_amp, mu, rho, gamma = generate_rand_tensors(1) #create rand tensors for excitation and medium
        model.run(dev, 1, dt, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, ex_amp, disp =True, dispRate = 1/1, pause = pause_sec)
        #run with B = 1
    else:
        ex_x, ex_y, ex_amp, mu, rho, gamma = generate_rand_tensors(B) 
        sol, sol_t = model.run(dev, B, dt, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, ex_amp)
        return sol, sol_t
    
    return


def generate_rand_tensors(_B):
    rd_x = torch.randint(0, w-2, (_B,)) 
    rd_y = torch.randint(0, h-2, (_B,))
    rd_amp = torch.randn(_B)
    rd_mu = torch.rand(_B) #these will prob need adjustment in order to match realistic values.
    rd_rho = torch.rand(_B)
    rd_gamma = torch.rand(_B)
    return rd_x, rd_y, rd_amp, rd_mu, rd_rho, rd_gamma
