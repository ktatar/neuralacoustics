import torch
import configparser # to save config in new ini file
import scipy.io # to save dataset
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time
from neuralacoustics.utils import openConfig

N = -1
B = -1
w = -1
h = -1
mu = torch.empty((1,))
rho = torch.empty((1,))
gamma = torch.empty((1,))
seed = -1
nsteps = -1
dryrun = -1
dt = -1
pause_sec = -1
model = 0

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
    
    #open config file
    generator_name = Path(__file__).stem
    config = openConfig(config_path, generator_name)
    
    #------------------------------------------------------------------------------------------------------------------------------------
    #parameters
    
    #model
    num_model = config['dataset_generator_parameters'].get('numerical_model') #path of num_model
    num_model_name = Path(num_model).parts[-1] 
    
    num_model_path = Path(num_model.replace('PRJ_ROOT', prj_root)).joinpath(num_model_name +'.py') # complete path to file/model

    #model config file
    num_model_config_path = config['dataset_generator_parameters'].get('numerical_model_config')

    # default config has same name as generator and is in same folder
    if num_model_config_path == 'default' or num_model_config_path == '':
        num_model_config_path = Path(num_model.replace('PRJ_ROOT', prj_root)).joinpath(num_model_name +'.ini') # model_dir/model_name/model_name.ini
    elif num_model_config_path == 'this_file':
        num_model_config_path = config_path
    else:
        num_model_config_path = Path(num_model_config_path.replace('PRJ_ROOT', prj_root))

    # dataset size
    N = config['dataset_generator_parameters'].getint('N') # num of dataset points
    B = config['dataset_generator_parameters'].getint('B') # batch size
    
    # seconds to pause between datapoints during visualization
    pause_sec = pause
    # only used in dry run and it will be ignored in solver if <= 0
    
    #seed
    seed = config['dataset_generator_parameters'].getint('seed') #for determinism.
    
    # domain size
    w = config['numerical_model_parameters'].getint('w') # domain width [cells]
    h = config['numerical_model_parameters'].getint('h') # domain height[cells]

    mu[0] = config['numerical_model_parameters'].getfloat('mu') 
    rho[0] = config['numerical_model_parameters'].getfloat('rho') 
    gamma[0] = config['numerical_model_parameters'].getfloat('gamma')
    
    # time parameters
    nsteps = config['numerical_model_parameters'].getint('nsteps') # = T_in+T_out, e.g., Tin = 10, T = 10 -> input [0,Tin), output [10, Tin+T)
    dt = 1.0 / config['numerical_model_parameters'].getint('samplerate') # seconds (1/Hz), probably no need to ever modify this...
    

    #------------------------------------------------------------------------------------------------------------------------------------
    #loads model    
    model_path_folders = num_model_path.parts
    # create package structure by concatenating folders with '.'
    packages_struct = '.'.join(model_path_folders)[:-3] # append all parts and remove '.py' from file/package name
    
    # import and load 
    model = __import__(packages_struct, fromlist=['load, run']) # model.path.numerical_model is model script [i.e., package]  
    model.load(num_model_config_path, prj_root) #loads solver for model
    
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
        
    rem = 0 # is there any remainder?
    
    return num_of_batches, ch, rem, N, B, h, w, nsteps, dt, num_model_config_path

def generate_datasetBatch(dev, dryrun):
    if dryrun == 0:
        ex_x, ex_y, noise_submatrix = generate_fullNoiseMatrix(B) 
        sol, sol_t = model.run(dev, B, dt, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, noise_submatrix)
    else:
        ex_x, ex_y, noise_submatrix = generate_fullNoiseMatrix(1) #create rand tensors for excitation and medium
        sol, sol_t = model.run(dev, 1, dt, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, noise_submatrix, disp =True, dispRate = 1/1, pause = pause_sec) #run with B = 1
    return sol, sol_t


def generate_fullNoiseMatrix(_B):
    _ex_x = torch.zeros(_B) 
    _ex_y = torch.zeros(_B)
    _noise_matrix = torch.randn(_B, h-2, w-2)
    return _ex_x, _ex_y, _noise_matrix

def getSolverInfoFromModel():
    return model.getSolverInfo()

