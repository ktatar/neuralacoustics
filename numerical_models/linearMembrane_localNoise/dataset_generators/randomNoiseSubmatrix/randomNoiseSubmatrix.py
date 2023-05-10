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
init_size_min = -1
init_size_max = -1
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
    global init_size_min
    global init_size_max 
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
    
    # noise submatrix side (using eval to allow the use of w and h), these must be ints.
    init_size_min = int(eval(config['dataset_generator_parameters'].get('init_size_min'))) # minimum side of noise submatrix [cells]
    init_size_max = int(eval(config['dataset_generator_parameters'].get('init_size_max'))) # maximum side of noise submatrix [cells]
    init_size_min, init_size_max = check_inputs(init_size_min, init_size_max)


    #------------------------------------------------------------------------------------------------------------------------------------
    # imports + loads model

    model_function_list = ['load, run']  # specify which functions to load.
    model, model_config_path = import_fromScript(prj_root, config_path, model_path, model_config_path, function_list=model_function_list)

    n_sol = model.load(model_config_path, prj_root)  # loads solver for model

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

    return num_of_batches, ch, N, B, h, w, nsteps, dt, model_config_path, n_sol


def generate_datasetBatch(dev, dryrun):
    if dryrun == 0:
        rd_x, rd_y, rand_size = generate_randNoiseParams(B) 
        full_excitation, sol = model.run(dev, B, dt, nsteps, w, h, mu, rho, gamma, rd_x, rd_y, rand_size, rand_size)
    else:
        rd_x, rd_y, rand_size = generate_randNoiseParams(1) #create rand tensors for excitation and medium
        full_excitation, sol = model.run(dev, 1, dt, nsteps, w, h, mu, rho, gamma, rd_x, rd_y, rand_size, rand_size, disp=True, dispRate=1/1, pause=pause_sec) #run with B = 1
    return full_excitation, sol


def generate_randNoiseParams(_B):
    # generate random sizes for the submatrix, min & max sizes both inclusive
    rand_size = torch.randint(init_size_min, init_size_max+1, (_B,), dtype=torch.long)

    # create random indices for the submatrix
    rd_y = torch.randint(0, h-2, (B,)) 
    max_indices_h = h-2-rand_size  # max indices possible for each submatrix size (domain size-2(for the rim)-submatrix size)
    rd_y = torch.remainder(rd_y, max_indices_h+1)  # take elementwise index_value % (max_index_value+1), keeps indices in range
    rd_y += 1 # add one to shift indices into center

    rd_x = torch.randint(0, w-2, (B,))  # repeat for x-axis
    max_indices_w = w-2-rand_size
    rd_x = torch.remainder(rd_x, max_indices_w+1)
    rd_x += 1

    return rd_x, rd_y, rand_size


def check_inputs(init_size_min, init_size_max):
    min_default = 1
    max_default = min(h, w)-2

    # checking values for init_size min
    if init_size_min < min_default:
        init_size_min = min_default
        print(f"Value for init_size_min is less than {min_default}, will be clipped to {min_default}")
    elif init_size_min > max_default:
        init_size_min = max_default
        print(f"Value for init_size_min is greater than {max_default}, will be clipped to {max_default}")

    # checking values for init_size_max
    if init_size_max < min_default:
        init_size_max = min_default
        print(f"Value for init_size_max is less than {min_default}, will be clipped to {min_default}")
    elif init_size_max > max_default:
        init_size_max = max_default
        print(f"Value for init_size_max is greater than {max_default}, will be clipped to {max_default}")

    # checking that min < max
    if init_size_min > init_size_max:
        init_size_min = init_size_max
        print(f"Value for init_size_min is greater than init_size_max, will be clipped to {init_size_max}")

    return init_size_min, init_size_max


def getSolverInfoFromModel():
    return model.getSolverInfo()
