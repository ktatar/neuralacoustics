import torch
import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os
from neuralacoustics.dataset_loader import loadDataset # to load dataset
from neuralacoustics.data_plotter import plotDomain # to plot data entries (specific series of domains)


# retrieve PRJ_ROOT
prj_root = Path(__file__).absolute() # path of this script, which is in PRJ_ROOT
prj_root = prj_root.relative_to(Path.cwd()) # path of this script, relative to the current working directory, i.e, from where the script was called 
prj_root = str(prj_root.parent) # dir of this script, relative to working dir (i.e, PRJ_ROOT)


#-------------------------------------------------------------------------------
# simulation parameters

# Parse command line arguments
parser = argparse.ArgumentParser()
default_config = str(Path(prj_root).joinpath('default.ini'))
parser.add_argument('--config', type=str, default =default_config , help='path of config file')
args = parser.parse_args()

# Get config file
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)

try:
  with open(config_path) as f:
      config.read_file(f)
except IOError:
    print(f'dataset_visualizer: Config file not found --- \'{config_path}\'')
    quit()




#-------------------------------------------------------------------------------
# read params from config file

# dataset name
dataset_name = config['dataset'].get('name')
# dataset dir
dataset_dir = config['dataset'].get('dir')
dataset_dir = dataset_dir.replace('PRJ_ROOT', prj_root)


# total number of data points to load, i.e., specific sub-series of time steps within data entries
n = config['dataset'].getint('n_train') + config['dataset'].getint('n_test') 

# size of window, i.e., length of each data point
window = config['dataset'].getint('window_size') 

# offset between consecutive windows
stride = config['dataset'].getint('window_stride') 
# by default, windows are juxtaposed
if stride <= 0:
    stride = window

# maximum index of the frame (timestep) that can be retrieved from each dataset entry
limit = config['dataset'].getint('window_limit') 


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
print(f'\tconsecutive datapoints are {stride} timsteps apart -> window_stride')
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


