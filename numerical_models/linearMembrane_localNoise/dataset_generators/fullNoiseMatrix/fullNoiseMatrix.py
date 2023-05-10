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

    # model path
    model_path = config['dataset_generator_parameters'].get('numerical_model')  # path of num_model

    # model config file
    model_config_path = config['dataset_generator_parameters'].get('numerical_model_config')

    # dataset size
    N = config['dataset_generator_parameters'].getint('N') # num of dataset points
    B = config['dataset_generator_parameters'].getint('B') # batch size
    
    # seconds to pause between datapoints during visualization
    pause_sec = pause
    # only used in dry run and it will be ignored in solver if <= 0
    
    # seed
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
    # imports + loads model

    model_function_list = ['load, run']  # specify which functions to load.
    model, model_config_path = import_fromScript(prj_root, config_path, model_path, model_config_path, function_list=model_function_list)

    n_sol = model.load(model_config_path, prj_root)  # loads solver for model
    
    #-----------------------------------------------------------------------------------------------------------------------------------
    torch.use_deterministic_algorithms(True)  # enables determinism.
    torch.backends.cudnn.deterministic = True 
    torch.manual_seed(seed)  # sets seed
    
    #----------------------------------------------------------------------------
    #compute meta data, e.g., duration, actual size...

    num_of_batches = N//B
    
    # num of chunks must be lower than total number of batches
    if ch > num_of_batches:
      ch = num_of_batches//2 # otherwise, a chunk every other batch
    if ch == 0: # always at least one chunk!
      ch = 1

    return num_of_batches, ch, N, B, h, w, nsteps, dt, model_config_path, n_sol


def generate_datasetBatch(dev, dryrun):
    if dryrun == 0:
        rd_x, rd_y, rand_size_x, rand_size_y  = generate_fullNoiseParams(B) 
        full_excitation, sol = model.run(dev, B, dt, nsteps, w, h, mu, rho, gamma, rd_x, rd_y, rand_size_x, rand_size_y)
    else:
        rd_x, rd_y, rand_size_x, rand_size_y = generate_fullNoiseParams(1) #create rand tensors for excitation and medium
        full_excitation, sol = model.run(dev, 1, dt, nsteps, w, h, mu, rho, gamma, rd_x, rd_y, rand_size_x, rand_size_y, disp=True, dispRate=1/1, pause=pause_sec) #run with B = 1
    return full_excitation, sol


def generate_fullNoiseParams(_B):
    rd_x = torch.ones(_B, dtype=torch.long) #always at upper left corner, makes sure we're using
    rd_y = torch.ones(_B, dtype=torch.long) 
    rand_size_x = torch.ones(_B, dtype=torch.long) * (w-2) 
    rand_size_y = torch.ones(_B, dtype=torch.long) * (h-2) 
    return rd_x, rd_y, rand_size_x, rand_size_y


def getSolverInfoFromModel():
    return model.getSolverInfo()

