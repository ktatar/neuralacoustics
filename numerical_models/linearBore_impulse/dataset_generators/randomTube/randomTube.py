import torch
from pathlib import Path # to properly handle paths and folders on every os
from neuralacoustics.utils import openConfig
from neuralacoustics.utils import import_file

N = -1
B = -1
w = -1
h = -1
c = torch.empty((1,)) # declared as tensor to facilitate vector operations
rho = torch.empty((1,))
mu = torch.empty((1,))
seed = -1
nsteps = -1
dryrun = -1
dt = -1
pause_sec = -1
model = 0
tube_len_min = -1
tube_len_max = -1
tube_width_min = -1
tube_width_max = -1


def load(config_path, ch, prj_root, pause):
    # in same style as load_test, this function writes the config variables to global variables.
    global N 
    global B 
    global w 
    global h
    global mu
    global rho
    global c
    global nsteps 
    global dt
    global dryrun 
    global pause_sec
    global model
    global tube_len_min 
    global tube_len_max 
    global tube_width_min
    global tube_width_max 
    
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
    
    #seed
    seed = config['dataset_generator_parameters'].getint('seed') #for determinism.
    
    # tube length range (using eval to allow the use of w and h), these must be ints.
    tube_len_min = int(eval(config['dataset_generator_parameters'].get('tube_len_min'))) # minimum length of tube [cells]
    tube_len_max = int(eval(config['dataset_generator_parameters'].get('tube_len_min'))) # maximum length of tube[cells]
    
    # tube length range (using eval to allow the use of w and h), these must be ints.
    tube_width_min = int(eval(config['dataset_generator_parameters'].get('tube_width_min'))) # minimum width of tube [cells]
    tube_width_max = int(eval(config['dataset_generator_parameters'].get('tube_width_max'))) # maximum width of tube[cells]
    
    # domain size
    w = config['numerical_model_parameters'].getint('w') # domain width [cells]
    h = config['numerical_model_parameters'].getint('h') # domain height[cells]

    mu[0] = config['numerical_model_parameters'].getfloat('mu') 
    rho[0] = config['numerical_model_parameters'].getfloat('rho') 
    c[0] = config['numerical_model_parameters'].getfloat('c')
    
    # time parameters
    nsteps = config['numerical_model_parameters'].getint('nsteps') # = T_in+T_out, e.g., Tin = 10, T = 10 -> input [0,Tin), output [10, Tin+T)
    dt = 1.0 / config['numerical_model_parameters'].getint('samplerate') # seconds (1/Hz), probably no need to ever modify this...


    #------------------------------------------------------------------------------------------------------------------------------------
    # imports + loads model

    model_function_list = ['load, run']  # specify which functions to load.
    model, model_config_path = import_file(prj_root, config_path, model_path, model_config_path, function_list=model_function_list)

    model.load(model_config_path, prj_root)  # loads solver for model
    
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
    
    return num_of_batches, ch, N, B, h, w, nsteps, dt, model_config_path

def generate_datasetBatch(dev, dryrun):
    if dryrun == 0:
        tube_x, tube_y, tube_length, tube_width, ex_mag = generate_randTubeParams(B)
        sol, sol_t = model.run(dev, B, dt, nsteps, w, h, mu, rho, c, tube_x, tube_y, tube_length, tube_width, ex_mag)
    else:
        tube_x, tube_y, tube_length, tube_width, ex_mag = generate_randTubeParams(1) #create rand tensors for excitation and medium
        sol, sol_t = model.run(dev, 1, dt, nsteps, w, h, mu, rho, c, tube_x, tube_y, tube_length, tube_width, ex_mag, disp =True, dispRate = 1/1, pause = pause_sec) #run with B = 1
    return sol, sol_t


def generate_randTubeParams(_B):
    
    #generate x-axis parameters
    tube_x = torch.randint(0, w, (B,)) # random indices (across full domain)
    tube_len = torch.randint(tube_len_min, tube_len_max + 1, (_B, ), dtype=torch.long)# generates a random length, between min_side and max_side (both inclusive)
    max_indices_w = w - tube_len # max indices possible for a given tube length
    tube_x = torch.remainder(tube_x, max_indices_w + 1)# take elementwise index_value % (max_index_value+1), clips to [0,max_index_value]

    # repeat for y-axis
    tube_y = torch.randint(0, h, (B,)) # random indices (across full domain)
    tube_width = torch.randint(tube_width_min, tube_width_max + 1, (_B, ), dtype=torch.long)
    max_indices_h = h - (tube_width + 1) # bc of how we construct the walls of the tube, this needs to be offset by 1
    tube_y = torch.remainder(tube_y, max_indices_h) + 1 # clips to [1, max_index_value]
    
    ex_mag = torch.randn(_B)# excitation magnitude

    return tube_x, tube_y, tube_len, tube_width, ex_mag

def getSolverInfoFromModel():
    return model.getSolverInfo()

