import torch
from neuralacoustics.dataset_loader import loadDataset # to load dataset
from neuralacoustics.data_plotter import plotDomain # to plot data entries (specific series of domains)
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser


# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)


#-------------------------------------------------------------------------------
# simulation parameters

# get config file
config = getConfigParser(prj_root, __file__) # we call this script from command line directly
# hence __file__ is not a path, just the file name with extension

# read params from config file

# first how to load dataset

# dataset name
dataset_name = config['dataset_visualization'].get('dataset_name')
# dataset dir
dataset_dir = config['dataset_visualization'].get('dataset_dir')
dataset_dir = dataset_dir.replace('PRJ_ROOT', prj_root)

# total number of data points to load, i.e., specific sub-series of time steps within data entries
n = config['dataset_visualization'].getint('n_load') 

# in case we want to skip some chunk files to access points that are further aways without filling up all RAM
start_ch = config['dataset_visualization'].getint('start_ch') 

# size of window, i.e., length of each data point
window = config['dataset_visualization'].getint('window_size')
# zero means all available time steps, it is handled automatically in loadDataset() 

# offset between consecutive windows
stride = config['dataset_visualization'].getint('window_stride') 
# by default, windows are juxtaposed, so zero means step = window
# this is done automatically in loadDataset(), but we want to print result here too, so ti will be done again after loadDataset() is called

# maximum index of the frame (timestep) that can be retrieved from each dataset entry
limit = config['dataset_visualization'].getint('window_limit') 

# permute dataset entries
permute = config['dataset_visualization'].getint('permute') 
permute = bool(permute>0)

# then the actual visualization part

# number of datapoints to visualize
num_of_datapoints = config['dataset_visualization'].getint('n_visualize') 
# default is all
if num_of_datapoints <= 0:
    num_of_datapoints = n

# index of first datapoint to visualize
datapoint_index = config['dataset_visualization'].getint('first_datapoint') 

# number of time steps to plot from each visualized datapoint
timestep_range = config['dataset_visualization'].getint('timestep_range') 
# zero means all available time steps, it will be handled later, after loadDataset() is called

# seconds to pause between datapoints during visualization
pause = config['dataset_visualization'].getint('pause_sec')
# it will be ignored if <= 0

# misc
seed = config['dataset_visualization'].getint('seed') 
# for permutation determinism
torch.manual_seed(seed)

#-------------------------------------------------------------------------------
# retrieve all data points
u = loadDataset(dataset_name, dataset_dir, n, window, stride, limit, start_ch, permute)

shape = list(u.shape)

# actual values used in loadDataset()
window = shape[-1]
if stride <= 0:
    stride = window
if timestep_range <= 0:
    timestep_range = window

print('dataset shape:', shape)
print(f'\tdataset has {shape[0]} datapoints -> n')
print(f'\teach composed of {window} timsteps -> window_size')
print(f'\tconsecutive datapoints are {stride} timesteps apart -> window_stride')
if limit > 0:
  print(f'\tand there cannot be more than {limit} consecutive timsteps -> window_limit')
else :
  print('\tand no limit on consecutive timesteps -> window_limit')


#-------------------------------------------------------------------------------
# visualize

# when stride is lower than window size, it is still possible to visualize consecutive timsteps
# by setting timestep_range lower than window size
# this is handy when we want to visually check dataset entries

datapoints = u[datapoint_index:datapoint_index+num_of_datapoints, ...]
for d_n in range(0, num_of_datapoints):
    for t_n in range(0, timestep_range):
        print(f'datapoint {datapoint_index+d_n}, timstep {t_n} (max: {datapoints[d_n, ..., t_n].max()})')
        if pause <= 0:
          plotDomain(datapoints[d_n, ..., t_n])
        else:
          plotDomain(datapoints[d_n, ..., t_n], pause=pause)


