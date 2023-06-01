import torch
import configparser # to save config in new ini file
import scipy.io # to save dataset
import math
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time
from datetime import datetime
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.utils import openConfig

# for audio file
#import torchaudio #VIC may not work on some systems, e.g., M1 Macs
import numpy as np
from scipy.io.wavfile import write

#most of this is lifted directly from the dataset_generator file

# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)

#-------------------------------------------------------------------------------
# simulation parameters

# get config file
config, config_path = getConfigParser(prj_root, __file__) # we call this script from command line directly
# hence __file__ is not a path, just the file name with extension

# read params from config file, numerical_model_test section

# model
numerical_model_ = config['numerical_model_test'].get('numerical_model') # keep original string for dataset config
numerical_model_name = Path(numerical_model_).parts[-1]

print("Model name: ", numerical_model_name)
print("Model directory: ", numerical_model_) #added printout

model_path = Path(numerical_model_.replace('PRJ_ROOT', prj_root)).joinpath(numerical_model_name+'.py') # actual path to file/model

# model config file
model_config_path = config['numerical_model_test'].get('numerical_model_config')
# default config has same name as model and is in same folder
if model_config_path == 'default' or model_config_path == '':
  model_config_path = Path(numerical_model_.replace('PRJ_ROOT', prj_root)).joinpath(numerical_model_name+'.ini') # model_dir/model_name_.ini 
else:
  model_config_path = model_config_path.replace('PRJ_ROOT', prj_root)
  model_config_path = Path(model_config_path)

#-------------------------------------------------------------------------------

# load model
# we want to load the package through potential subfolders
model_path_folders = model_path.parts # creates full paths and gets folders
packages_struct = '.'.join(model_path_folders)[:-3] # append all parts and remove '.py' from file/package name

# import 
model = __import__(packages_struct, fromlist=['load_test', 'run_test']) # model.path.model_name_ is model script [i.e., package]

#-------------------------------------------------------------------------------

dev = config['numerical_model_test'].get('dev') # cpu or gpu

# in case of generic gpu or cuda explicitly, check if available
if dev == 'gpu' or 'cuda' in dev:
  if torch.cuda.is_available():  
    dev = torch.device('cuda')
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
  else:  
    dev = torch.device('cpu')
    print('numerical_model_test: gpu not avaialable!')

print('Device:', dev)

#----------------------------------------------------------------------------

disp_rate = 1/1
plot = config['numerical_model_test'].getint('plot')
pause_sec = config['numerical_model_test'].getfloat('pause_sec') #seconds to pause in between sim.

audio = config['numerical_model_test'].getint('audio')

# we need at least plot on
if plot == 0 and audio == 0:
    plot = 1

if audio:
  mic_x = config['numerical_model_test'].getint('mic_x')
  mic_y = config['numerical_model_test'].getint('mic_y')
  
  config = openConfig(model_config_path, __file__)
  w = config['numerical_model_parameters'].getint('w')
  h = config['numerical_model_parameters'].getint('h')
  if mic_x >= w or mic_y >= h:
    raise AssertionError("mic_x/mic_y out of bound")

  srate = config['numerical_model_parameters'].getint('samplerate')

    


model.load_test(model_config_path, prj_root)
if plot:
  sol, _ = model.run_test(dev, True, dispRate = disp_rate, pause = pause_sec)
else:
  sol, _ = model.run_test(dev, False)

if audio:
  samples = sol[0, mic_y, mic_x, :]
  
  current_time = datetime.now()
  current_time = current_time.strftime('%Y-%m-%d_%H:%M:%S')
  file_name = f"{numerical_model_name}_{current_time}.wav"

  
  
  #samples = torch.clamp(samples, -1, 1)

  #samples = samples.unsqueeze(0)
  #torchaudio.save(file_name, samples, math.floor(1/dt))
  
  samples = samples.cpu().detach().numpy()
  write(file_name, srate, samples)

