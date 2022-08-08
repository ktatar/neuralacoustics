import torch
import configparser # to save config in new ini file
import scipy.io # to save dataset
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.utils import openConfig


prj_root = getProjectRoot(__file__)


model_name_ = 'num_model_3'
gen_name_ = 'dataset_generator_0'
model_root_ = 'PRJ_ROOT/numerical_models'
# load generator.

# model
model_root = model_root_.replace('PRJ_ROOT', prj_root)
model_root = Path(model_root)
model_dir = model_root.joinpath(model_name_) # model_dir = model_root/model_name_ -> it is folder, where model script and its config file reside
model_config_path = model_dir.joinpath(gen_name_+'.ini') # model_dir/model_name_.ini 

#taken from original dataset generator code.
# we want to load the package through potential subfolders
# we can pretend we are in the PRJ_ROOT, for __import__ will look for the package from there
model_path_folders = Path(model_root_.replace('PRJ_ROOT', '.')).joinpath(model_name_).parts # also add folder with same name as model

# create package structure by concatenating folders with '.'
packages_struct = model_path_folders[0]
for pkg in range(1,len(model_path_folders)):
    packages_struct += '.'+model_path_folders[pkg] 
# load 
gen = __import__(packages_struct + '.' + gen_name_, fromlist=['*']) # model.path.model_name_ is model script [i.e., package]

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True 
torch.manual_seed(0)#used if you want the same events on successive running of the scripts, but diff results when running the function the same time.

gen.load_generator(model_config_path, prj_root)
gen.generate_dataset()
gen.generate_dataset()
