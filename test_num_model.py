import torch
import configparser # to save config in new ini file
import scipy.io # to save dataset
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.utils import openConfig

#most of this is lifted directly from the dataset_generator file

# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)

#-------------------------------------------------------------------------------
# simulation parameters

# get config file
config = getConfigParser(prj_root, __file__) # we call this script from command line directly
# hence __file__ is not a path, just the file name with extension

# read params from config file 
# uses numerical_model_test section instead of dataset_generation section


# model
model_root_ = config['numerical_model_test'].get('numerical_model_dir') # keep original string for dataset config
model_root = model_root_.replace('PRJ_ROOT', prj_root)
model_root = Path(model_root)
model_name_ = config['numerical_model_test'].get('numerical_model')
model_dir = model_root.joinpath(model_name_) # model_dir = model_root/model_name_ -> it is folder, where model script and its config file reside

print("Model name: ", model_name_)
print("Model directory: ", model_dir) #added printout

# model config file
model_config_path = config['numerical_model_test'].get('numerical_model_config')
# default config has same name as model and is in same folder
if model_config_path == 'default' or model_config_path == '':
  model_config_path = model_dir.joinpath(model_name_+'.ini') # model_dir/model_name_.ini 
else:
  model_config_path = model_config_path.replace('PRJ_ROOT', prj_root)
  model_config_path = Path(model_config_path)

#-------------------------------------------------------------------------------

# load model
# we want to load the package through potential subfolders
# we can pretend we are in the PRJ_ROOT, for __import__ will look for the package from there
model_path_folders = Path(model_root_.replace('PRJ_ROOT', '.')).joinpath(model_name_).parts # also add folder with same name as model

# create package structure by concatenating folders with '.'
packages_struct = model_path_folders[0]
for pkg in range(1,len(model_path_folders)):
    packages_struct += '.'+model_path_folders[pkg] 
# load 
model = __import__(packages_struct + '.' + model_name_, fromlist=['*']) # model.path.model_name_ is model script [i.e., package]

#-------------------------------------------------------------------------------

dev = config['dataset_generation'].get('dev') # cpu or gpu

# in case of generic gpu or cuda explicitly, check if available
if dev == 'gpu' or 'cuda' in dev:
  if torch.cuda.is_available():  
    dev = torch.device('cuda')
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
  else:  
    dev = torch.device('cpu')
    print('dataset_generator: gpu not avaialable!')

print('Device:', dev)

#----------------------------------------------------------------------------

disp_rate = 1/1
pause_sec = config['numerical_model_test'].getfloat('pause_sec') #seconds to pause in between sim.

model.load_test(model_config_path)
model.run_test(dev, dispRate = disp_rate ,pause = pause_sec)
