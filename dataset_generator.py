import torch
import configparser, argparse # to read config from ini file
import scipy.io # to save dataset
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time


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
    print('dataset_generator: Config file not found --- \'{}\''.format(config_path))
    quit()


#-------------------------------------------------------------------------------
# read params from config file

# model
model_root_ = config['dataset_generation'].get('numerical_model_dir') # keep original string for log
model_root = model_root_.replace('PRJ_ROOT', prj_root)
model_root = Path(model_root)
model_name_ = config['dataset_generation'].get('numerical_model')
model_dir = model_root.joinpath(model_name_) # model_dir = model_root/model_name_ -> it is folder, where model script and its config file reside

# model config file
model_config_path = config['dataset_generation'].get('numerical_model_config')
# default config has same name as model and is in same folder
if model_config_path == 'default' or model_config_path == '' :
  model_config_path = model_dir.joinpath(model_name_+'.ini') # model_dir/model_name_.ini 
else :
  model_config_path = model_config_path.replace('PRJ_ROOT', prj_root)
  model_config_path = Path(model_config_path)


# dataset size
N = config['dataset_generation'].getint('N') # num of dataset entries
B = config['dataset_generation'].getint('B') # batch size

# domain size
w = config['dataset_generation'].getint('w') # domain width [cells]
h = config['dataset_generation'].getint('h') # domain height[cells]

# time parameters
nsteps = config['dataset_generation'].getint('nsteps') # = T_in+T_out, e.g., Tin = 10, T = 10 -> input [0,Tin), output [10, Tin+T)
samplerate = config['dataset_generation'].getint('samplerate'); # Hz, probably no need to ever modify this...

# chunks
ch = config['dataset_generation'].getint('chunks') # num of chunks

# dataset dir
dataset_root_ = config['dataset_generation'].get('dataset_dir') # keep original string for log
dataset_root = dataset_root_.replace('PRJ_ROOT', prj_root)
dataset_root = Path(dataset_root)

dryrun = config['dataset_generation'].getint('dryrun') # visualize a single simulation run or save full dataset

#-------------------------------------------------------------------------------




# load model
# we want to load the package through potential subfolders
# we can pretend we are in the PRJ_ROOT, for __import__ will look for the package from there
model_path_folders = Path(model_root_.replace('PRJ_ROOT', '.')).joinpath(model_name_).parts # also add folder with same name as model

# create package structure by concatenating folders with '.'
packages_struct = model_path_folders[0]
for pkg in range(1,len(model_path_folders)) :
    packages_struct += '.'+model_path_folders[pkg] 
# load 
model = __import__(packages_struct + '.' + model_name_, fromlist=['*']) # model.path.model_name_ is model script [i.e., package]


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"

print('device:', dev)



# either generate full dataset and save it
if dryrun == 0 :

  # compute name of dataset

  # count datastes in folder 
  datasets = list(Path(dataset_root).glob('*'))
  num_of_datasets = len(datasets)
  # choose new dataset index accordingly
  DATASET_INDEX = str(num_of_datasets)

  name_clash = True

  while name_clash :
    name_clash = False
    for dataset in datasets :
      # in case a dataset with same name is there
      if Path(dataset).parts[-1] == 'dataset_'+DATASET_INDEX :
        name_clash = True
        DATASET_INDEX = str(int(DATASET_INDEX)+1) # increase index

  dataset_name = 'dataset_'+DATASET_INDEX
  dataset_dir = dataset_root.joinpath(dataset_name)



  #-------------------------------------------------------------------------------
  # compute meta data, e.g., duration, actual size...
  
  time_duration = nsteps/samplerate 
  print('simulation duration: ', time_duration, 's')

  # num of chunks must be lower than total number of batches
  if ch > N//B :
    ch = (N//B)//2 # a chunk every other batch
  if ch == 0 :
    ch = 1

  
  # it is possible that not all requested points are generated,
  # depending on ratio with requested batch size
  # this is why we calculate the actual_size of the dataset once saved
  n_cnt=0
  num_of_batches = N//B
  batches_per_ch = num_of_batches//ch
  ch_size = batches_per_ch * B # num of data points per chunk
  ch_cnt = 0
  rem = 0 # is there any remainder?

  # create folder where to save dataset
  dataset_dir.mkdir(parents=True, exist_ok=True)


  # compute number of leading zeros for pretty file names
  ch_num = str(ch)
  l_zeros=len(ch_num) # number of leading zeros in file name
  # check if l_zeros needs to be lowered down
  # e.g., ch_num = 100 -> [0, 99] -> should be printed with only one leading zero:
  # 01, 02, ..., 98, 99
  cc = pow(10,l_zeros-1)
  if ch <= cc :
    l_zeros = l_zeros-1 



  #-------------------------------------------------------------------------------
  # compute full dataset and save it

  t1 = default_timer()

  # initial conditions
  a = torch.zeros(ch_size, h, w)
  # solutions
  u = torch.zeros(ch_size, h, w, nsteps)


  for b in range(num_of_batches):
    # compute all steps in full batch
    sol, sol_t, p0 = model.run(dev, 1/samplerate, nsteps, B, w, h, model_name_, model_config_path)
    
    # store
    a[n_cnt:(n_cnt+B),...] = p0 # initial condition
    u[n_cnt:(n_cnt+B),...] = sol # results

    n_cnt += B

    # save some chunk, just in case...
    if (b+1) % batches_per_ch == 0 : 
      file_name = dataset_name + '_ch' + str(ch_cnt).zfill(l_zeros) + '_' + str(n_cnt) + '.mat'
      dataset_path = dataset_dir.joinpath(file_name)
      scipy.io.savemat(dataset_path, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
      print( '\tchunk {}, {} dataset points  (up to batch {} of {})'.format(ch_cnt, ch_size, b+1, num_of_batches) )
      ch_cnt += 1
      # reset initial conditions, solutions and data point count
      a = torch.zeros(ch_size, h, w)
      u = torch.zeros(ch_size, h, w, nsteps)
      n_cnt = 0
    elif (b+1) == num_of_batches :
      file_name = dataset_name + '_rem_' + str(n_cnt)  + '.mat'
      dataset_path = dataset_dir.joinpath(file_name)
      scipy.io.savemat(dataset_path, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
      print( '\tremainder, {} dataset points (up to batch {} of {})'.format(n_cnt, b+1, num_of_batches) )
      rem = 1

  t2 = default_timer()

  rem_size = rem*n_cnt
  actual_size = (ch_size * ch) + rem_size

  print('\nDataset', dataset_name, 'saved in:')
  print('\t', dataset_dir)
  print('total number of data points: {} (out of {} requested)'.format(actual_size, N))
  if ch == 1 :
    print('in a single chunk')
  else :
    print('split in', ch_cnt, 'chunks with', ch_size, 'datapoints each')
    if rem :
      print('plus remainder file with', n_cnt, 'datapoints')

  simulation_duration = t2-t1
  print('\nElapsed time:', simulation_duration, 's')



  # ------------------------------------------------------------------------------------------------

  # save relavant bits from general config file + extra info from model config file into a new dataset config file (log)

  # create empty config file
  config = configparser.RawConfigParser()
  config.optionxform = str # otherwise raw config parser converts all entries to lower case letters

  # fill it with dataset details
  config.add_section('dataset_details')
  config.set('dataset_details', 'name', dataset_name)
  config.set('dataset_details', 'actual_size', actual_size)
  config.set('dataset_details', 'simulated_time_s', time_duration)
  config.set('dataset_details', 'simulation_duration_s', simulation_duration)
  config.set('dataset_details', 'chunk_size', ch_size)
  config.set('dataset_details', 'remainder_size', rem_size)

  config.add_section('dataset_generation')
  config.set('dataset_generation', 'numerical_model_dir', model_root_)
  config.set('dataset_generation', 'numerical_model', model_name_)
  # the model config file is the current log file
  model_config_path_ = Path(dataset_root_).joinpath(dataset_name) # up to dataset folder, with PRJ_ROOT var
  dataset_name = dataset_name+'.ini'
  model_config_path_ = model_config_path_.joinpath(dataset_name) # add file name
  model_config_path_ = str(model_config_path_)
  config.set('dataset_generation', 'numerical_model_config', model_config_path_)
  config.set('dataset_generation', 'N', N)
  config.set('dataset_generation', 'B', B)
  config.set('dataset_generation', 'w', w)
  config.set('dataset_generation', 'h', h)
  config.set('dataset_generation', 'samplerate', samplerate)
  config.set('dataset_generation', 'nsteps', nsteps)
  config.set('dataset_generation', 'chunks', ch_cnt)
  config.set('dataset_generation', 'dataset_dir', dataset_root_)
  config.set('dataset_generation', 'dryrun', 0)


  # then retrieve model and solver details from model config file
  model_config = configparser.ConfigParser(allow_no_value=True)

  try:
      with open(model_config_path) as f:
        model_config.read(model_config_path)
  except IOError:
      print('dataset_generator: Model config file not found --- \'{}\''.format(model_config_path))
      sys.exit()

  # extract relevant bits and add them to new dataset config file
  config.add_section('numerical_model_details')
  for (each_key, each_val) in model_config.items('numerical_model_details') :
      config.set('numerical_model_details', each_key, each_val)

  config.add_section('solver')
  for (each_key, each_val) in model_config.items('solver') :
      config.set('solver', each_key, each_val)
  config.set('solver', 'description', model.getSolverDescription())
  
  config.add_section('numerical_model_parameters')
  for (each_key, each_val) in model_config.items('numerical_model_parameters') :
      config.set('numerical_model_parameters', each_key, each_val)

  # where to write it
  config_path = dataset_dir.joinpath(dataset_name)
  # write
  with open(config_path, 'w') as configfile:
      config.write(configfile)




else :
  # or generate 1 data entry and visualize it

  disp_rate = 1/1
  b=1 # 1 entry batch

  sol, _, _ = model.run(dev, 1/samplerate, nsteps, b, w, h, model_name_, model_config_path, True, disp_rate)
