import torch
import configparser # to save config in new ini file
import scipy.io # to save dataset
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.utils import openConfig

import torch
import configparser # to save config in new ini file
import scipy.io # to save dataset
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.utils import openConfig


# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)


#-------------------------------------------------------------------------------
# simulation parameters

# get config file
config = getConfigParser(prj_root, __file__) # we call this script from command line directly
# hence __file__ is not a path, just the file name with extension

# read params from config file

# model (same code for now because generator script is in the same model script).
model_root_ = config['dataset_generation'].get('numerical_model_dir') # keep original string for dataset config
model_root = model_root_.replace('PRJ_ROOT', prj_root)
model_root = Path(model_root)
model_name_ = config['dataset_generation'].get('numerical_model')
model_dir = model_root.joinpath(model_name_) # model_dir = model_root/model_name_ -> it is folder, where model script and its config file reside

generator_name = config['dataset_generation'].get('generator_name') #added to load generator script.

# model config file (tweaked so now it uses generator name)
model_config_path = config['dataset_generation'].get('numerical_model_config')
# default config has same name as model and is in same folder
if model_config_path == 'default' or model_config_path == '':
  model_config_path = model_dir.joinpath(generator_name +'.ini') # model_dir/model_name_.ini 
else:
  model_config_path = model_config_path.replace('PRJ_ROOT', prj_root)
  model_config_path = Path(model_config_path)


# chunks
ch = config['dataset_generation'].getint('chunks') # num of chunks

# dataset dir
dataset_root_ = config['dataset_generation'].get('dataset_dir') # keep original string for dataset config
dataset_root = dataset_root_.replace('PRJ_ROOT', prj_root)
dataset_root = Path(dataset_root)

dryrun = config['dataset_generation'].getint('dryrun') # visualize a single simulation run or save full dataset



#-------------------------------------------------------------------------------
# load model (generator)
#will need to tweak if generator file is moved to a diff folder(currently in numerical_models/num_model_3)

# we want to load the package through potential subfolders
# we can pretend we are in the PRJ_ROOT, for __import__ will look for the package from there
model_path_folders = Path(model_root_.replace('PRJ_ROOT', '.')).joinpath(model_name_).parts # also add folder with same name as model

# create package structure by concatenating folders with '.'
packages_struct = model_path_folders[0]
for pkg in range(1,len(model_path_folders)):
    packages_struct += '.'+model_path_folders[pkg] 
# load 
generator = __import__(packages_struct + '.' + generator_name, fromlist=['*']) # model.path.model_name_ is model script [i.e., package]

#-------------------------------------------------------------------------------

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True 
#torch.manual_seed(0)#used if you want the same events on successive running of the scripts, but diff results when running the function the same time.

generator.load_generator(model_config_path, prj_root)
generator.generate_dataset()
generator.generate_dataset()
