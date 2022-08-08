import torch
import configparser # to save config in new ini file
from pathlib import Path # to properly handle paths and folders on every os
from neuralacoustics.utils import openConfig

numerical_model = ''
numerical_model_dir = ''
numerical_model_config = ''
solver_dir = ''
solver_name = ''
N = -1
B = -1
w = -1
h = -1
seed = 0
nsteps = -1
dt = -1
dev = ''
pause_sec = -1
model = ''

def load_generator(config_path, prj_root):
    # in same style as load_test, this function writes the config variables to global variables.
    global numerical_model 
    global numerical_model_dir 
    global numerical_model_config
    global solver_dir
    global solver_name 
    global N 
    global B 
    global w 
    global h
    global nsteps 
    global dt
    global dev 
    global dryrun 
    global pause_sec
    global model
    
    model_name = Path(__file__).stem
    config = openConfig(config_path, model_name)
    
    #parameters
    numerical_model = config['dataset_generation'].get('numerical_model')
    numerical_model_dir = config['dataset_generation'].get('numerical_model_dir')
    numerical_model_config = config['dataset_generation'].get('numerical_model_config')
    solver_dir = config['dataset_generation'].get('solver_dir')
    solver_name = config['dataset_generation'].get('solver_name')
    N = config['dataset_generation'].getint('N')
    B = config['dataset_generation'].getint('B')
    w = config['dataset_generation'].getint('w')
    h = config['dataset_generation'].getint('h')
    seed = config['dataset_generation'].getint('h')
    nsteps = config['dataset_generation'].getint('nsteps')
    dt = 1.0 / config['dataset_generation'].getint('samplerate')
    dev = config['dataset_generation'].get('dev')
    if dev == 'gpu' or 'cuda' in dev:
        if torch.cuda.is_available():
            dev = torch.device('cuda')
            #print(torch.cuda.current_device())
            #print(torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            dev = torch.device('cpu')
            print('dataset_generator: gpu not avaialable!')
            
    pause_sec = config['dataset_generation'].getfloat('pause_sec')
    #mu = config['dataset_generation'].getfloat('mu')
    #rho = config['dataset_generation'].getfloat('rho')
    #gamma = config['dataset_generation'].getfloat('gamma')
    # model
    # this is taken from the dataset_generator code, loads the model.
    #inefficient to do this every function call, will need rework.
    model_root_ = numerical_model_dir
    model_name_ = numerical_model
    
    model_root = numerical_model_dir.replace('PRJ_ROOT', prj_root)
    model_root = Path(model_root)
    model_dir = model_root.joinpath(model_name_) # model_dir = model_root/model_name_ -> it is folder, where model script and its config file reside

    # model config file
    model_config_path = numerical_model_config
    # default config has same name as model and is in same folder
    if model_config_path == 'default' or model_config_path == '':
        model_config_path = model_dir.joinpath(model_name_+'.ini') # model_dir/model_name_.ini 
    else:
        model_config_path = model_config_path.replace('PRJ_ROOT', prj_root)
        model_config_path = Path(model_config_path)

    #getting model

    model_path_folders = Path(model_root_.replace('PRJ_ROOT', '.')).joinpath(model_name_).parts # also add folder with same name as model

    # create package structure by concatenating folders with '.'
    packages_struct = model_path_folders[0]
    for pkg in range(1,len(model_path_folders)):
        packages_struct += '.'+model_path_folders[pkg] 
    # load 
    model = __import__(packages_struct + '.' + model_name_, fromlist=['*']) # model.path.model_name_ is model script [i.e., package]

    
    return


def generate_dataset():
    ex_x, ex_y, ex_amp, mu, rho, gamma = generate_rand_tensors(w, h, B)
    print('ex_x: ', ex_x)
    print('ex_y: ', ex_y)
    print('ex_amp: ', ex_amp)
    print('mu: ', mu)
    print('rho: ', rho)
    print('gamma: ', gamma)
    
    model.load(solver_dir, solver_name)
    s, t = model.run(dev, B, dt, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, ex_amp, disp=True, dispRate=1, pause=1)
    
    for batch in range(B):
        for n in range(nsteps):
            print('batch: ', batch)
            print('n: ', n)
            print(s[batch, :, :, n])#all batches, first step. excite works, repeatable. now check mu/rho/gamma work.
    return s, t

def generate_rand_tensors(_w, _h, _B, _seed = 0):
    #torch.manual_seed(_seed) can be used to if you want the same tensors every time this is called.
    rd_x = torch.randint(0, _w-2, (_B,)) 
    rd_y = torch.randint(0, _h-2, (_B,))
    rd_amp = torch.randn(_B)
    rd_mu = torch.rand(_B) #these will prob need adjustment in order to match realistic values.
    rd_rho = torch.rand(_B)
    rd_gamma = torch.rand(_B)
    return rd_x, rd_y, rd_amp, rd_mu, rd_rho, rd_gamma
