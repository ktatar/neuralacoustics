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
config = getConfigParser(prj_root, __file__.replace('.py', ''))

# read params from config file

# dataset name
dataset_name = config['dataset_visualization'].get('dataset_name')
# dataset dir
dataset_dir = config['dataset_visualization'].get('dataset_dir')
dataset_dir = dataset_dir.replace('PRJ_ROOT', prj_root)


# total number of data points to load, i.e., specific sub-series of time steps within data entries
n = config['dataset_visualization'].getint('n_train') + config['dataset_visualization'].getint('n_test') 

# size of window, i.e., length of each data point
window = config['dataset_visualization'].getint('window_size') 

# offset between consecutive windows
stride = config['dataset_visualization'].getint('window_stride') 
# by default, windows are juxtaposed
if stride <= 0:
    stride = window

# maximum index of the frame (timestep) that can be retrieved from each dataset entry
limit = config['dataset_visualization'].getint('window_limit') 


# number of datapoints to visualize
num_of_datapoints = config['dataset_visualization'].getint('num_of_datapoints') 
# at least one datapoint
if num_of_datapoints <= 0:
    timestep_range = 1

# index of first datapoint to visualize
datapoint_index = config['dataset_visualization'].getint('first_datapoint') 

# number of time steps to plot from each visualized datapoint
timestep_range = config['dataset_visualization'].getint('timestep_range') 
# zero means all available time steps
if timestep_range <= 0:
    timestep_range = window


#-------------------------------------------------------------------------------
# retrieve all data points
u = loadDataset(dataset_name, dataset_dir, n, window, stride, limit)

print(u.max())


shape = list(u.shape)
print('dataset shape:', shape)
print(f'\tdataset has {shape[0]} datapoints -> n_train+n_test')
print(f'\teach composed of {shape[-1]} timsteps -> window_size')
print(f'\tconsecutive datapoints are {stride} timesteps apart -> window_stride')
if limit > 0:
  print(f'\tand there cannot be more than {limit} consecutive timsteps -> window_limit')
else :
  print('\tand no limit on consecutive timsteps -> window_limit')


#-------------------------------------------------------------------------------
# visualize

# when stride is lower than window size, it is still possible to visualize consecutive timsteps
# by setting timestep_range lower than window size
# this is handy when we want to visually check dataset entries

datapoints = u[datapoint_index:datapoint_index+num_of_datapoints, ...]
for d_n in range(0, num_of_datapoints):
    for t_n in range(0, timestep_range):
        print(f'datapoint {datapoint_index+d_n}, timstep {t_n} (max: {datapoints[d_n, ..., t_n].max()})')
        plotDomain(datapoints[d_n, ..., t_n])


