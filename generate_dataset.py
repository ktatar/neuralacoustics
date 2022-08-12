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

generator_name = config['dataset_generation'].get('generator_name') #name of generator file 

generator_root_ = config['dataset_generation'].get('generator_dir') # generator_root_ variable only used for generator_config_path now.
generator_root = generator_root_.replace('PRJ_ROOT', prj_root) 
generator_root = Path(generator_root) #path for generator root(just the numerical model root for now)

generator_model = config['dataset_generation'].get('generator_model')# this is the numerical model used for dataset generation.

generator_dir = generator_root.joinpath(generator_model) # model_dir = model_root/model_name_ -> it is folder, where model script and its config file reside
#right now, generator file and model file are stored in same directory.

# generator config file (basically just the model config file code with names changed.)
generator_config_path = config['dataset_generation'].get('generator_config')

# default config has same name as generator and is in same folder
if generator_config_path == 'default' or generator_config_path == '':
  generator_config_path = generator_dir.joinpath(generator_name +'.ini') # generator_dir/generator_name_.ini 
else:
  generator_config_path = generator_config_path.replace('PRJ_ROOT', prj_root)
  generator_config_path = Path(generator_config_path)

#-------------------------------------------------------------------------------
# load model (generator)
#will need to tweak if generator file is moved to a diff folder(currently in numerical_models/num_model_3)

# we want to load the package through potential subfolders
# we can pretend we are in the PRJ_ROOT, for __import__ will look for the package from there
generator_path_folders = Path(generator_root_.replace('PRJ_ROOT', '.')).joinpath(generator_model).parts # also add folder with same name as model
#should the above be replaced with just generator_root.joinpath(generator_model).parts? both seem to work

# create package structure by concatenating folders with '.'
packages_struct = generator_path_folders[0]
for pkg in range(1,len(generator_path_folders)):
    packages_struct += '.'+generator_path_folders[pkg] 
# load

generator = __import__(packages_struct + '.' + generator_name, fromlist=['*']) # model.path.model_name_ is model script [i.e., package].


#-------------------------------------------------------------------------------
torch.use_deterministic_algorithms(True) #enables determinism.
torch.backends.cudnn.deterministic = True 
#torch.manual_seed(0) #sets seed

#-------------------------------------------------------------------------------
dev = config['dataset_generation'].get('dev') # cpu or gpu, keep original for dataset config

# in case of generic gpu or cuda explicitly, check if available
if dev == 'gpu' or 'cuda' in dev:
  if torch.cuda.is_available():  
    dev = torch.device('cuda')
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
  else:  
    dev = torch.device('cpu')
    print('dataset_generator: gpu not available!')

print('Device:', dev)

#------------------------------------------------------------------------------

generator.load_generator(generator_config_path, prj_root) #passing prj_root to maintain original code structure.
generator.generate_dataset(dev) #generates dataset.
