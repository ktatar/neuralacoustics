import torch
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.utils import import_fromScript

#most of this is lifted directly from the dataset_generator file

# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)

#-------------------------------------------------------------------------------
# simulation parameters

# get config file
config, config_path = getConfigParser(prj_root, __file__) # we call this script from command line directly
# hence __file__ is not a path, just the file name with extension

# read params from the config file's numerical_model_test section 

# model
model_path = config['numerical_model_test'].get('numerical_model') # path to numerical model

# model config file
model_config_path = config['numerical_model_test'].get('numerical_model_config')

# seconds to pause in between sim.
pause_sec = config['numerical_model_test'].getfloat('pause_sec') 

# device
dev = config['numerical_model_test'].get('dev') # cpu or gpu

# display rate
disp_rate = 1/1 

#-------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------
# load and run model

model_function_list = ['load_test', 'run_test']  # specify which functions to load.
model, model_config_path = import_fromScript(prj_root, config_path, model_path, model_config_path, function_list=model_function_list)

model.load_test(model_config_path, prj_root)
model.run_test(dev, dispRate = disp_rate ,pause = pause_sec)
