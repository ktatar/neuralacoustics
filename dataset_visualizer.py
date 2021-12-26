import torch
import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os
from neuralacoustics.dataset_loader import loadDataset # to load dataset
from neuralacoustics.data_plotter import plotDomain # to plot data entries (specific series of domains)



# retrieve PRJ_ROOT
prj_root = Path(__file__).absolute() # path and name of this script, which is in PRJ_ROOT
prj_root = prj_root.relative_to(Path.cwd()) # path and name of current file, relative to the current working directory, i.e, from where the script was called 
prj_root = str(prj_root.parent) # path to current file (PRJ_ROOT), relative to working dir


#-------------------------------------------------------------------------------
# simulation parameters

# Parse command line arguments
parser = argparse.ArgumentParser()
default_config = str(Path(prj_root).joinpath('default.ini'))
parser.add_argument('--config', type=str, default =default_config , help='path to config file')
args = parser.parse_args()

# Get config file
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)

try:
  with open(config_path) as f:
      config.read_file(f)
except IOError:
    print('dataset_visualizer: Config file not found --- \'{}\''.format(config_path))
    quit()




#-------------------------------------------------------------------------------
# read params from config file

# dataset name
dataset_name = config['dataset'].get('name')
# dataset path
dataset_path = config['dataset'].get('path')
dataset_path = dataset_path.replace('PRJ_ROOT', prj_root)


# total number of data points to load, i.e., specific sub-series of frames within data entries
n = config['dataset'].getint('n_train') + config['dataset'].getint('n_test') 

# size of window, i.e., length of each data point
window = config['dataset'].getint('window_size') 

# offset between consecutive windows
stride = config['dataset'].getint('window_stride') 
# by default, windows are juxtaposed
if stride <= 0 :
    stride = window

# maximum index of the frame (timestep) that can be retrieved from each dataset entry
limit = config['dataset'].getint('window_limit') 


# number of datapoints to visualize
num_of_datapoints = config['dataset_visualization'].getint('num_of_datapoints') 

# index of first datapoint to visualize
datapoint_index = config['dataset_visualization'].getint('first_datapoint') 

# number of frames to plot from each visualized datapoint
frame_range = config['dataset_visualization'].getint('frame_range') 
# zero means all available frames
if frame_range <= 0 :
    frame_range = window


#-------------------------------------------------------------------------------
# retrieve all data points
u = loadDataset(dataset_name, dataset_path, n, window, stride, limit)
shape = list(u.shape)
print('dataset shape:', shape)
print('\tdataset has', shape[0], 'datapoints -> n_train+n_test')
print('\teach composed of', shape[-1], 'frames (timsteps) -> window_size')
print('\tconsecutive datapoints are', stride, 'frames apart -> window_stride')
if limit > 0 :
  print('\tand there cannot be more than', limit, 'consecutive frams -> window_limit')
else :
  print('\tand no limit on consecutive frames -> window_limit')


#-------------------------------------------------------------------------------
# visualize

# when stride is lower than window size, it is still possible to visualize consecutive frames
# by setting frame_range lower than window size
# this is handy when we want to visually check dataset entries

datapoints = u[datapoint_index:datapoint_index+num_of_datapoints, ...]
for d_n in range(0, num_of_datapoints) :
    for f_n in range(0, frame_range) :
        print('datapoint', d_n, 'frame', f_n)
        plotDomain(datapoints[d_n, ..., f_n])


