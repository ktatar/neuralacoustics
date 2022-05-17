import torch
import configparser # to save config in new ini file
import scipy.io # to save dataset
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser

# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)

#-------------------------------------------------------------------------------
# simulation parameters

# get config file
config = getConfigParser(prj_root, __file__.replace('.py', ''))

# read params from config file

# model
model_root_ = config['dataset_generation'].get('numerical_model_dir') # keep original string for dataset config
model_root = model_root_.replace('PRJ_ROOT', prj_root)
model_root = Path(model_root)
model_name_ = config['dataset_generation'].get('numerical_model')
model_dir = model_root.joinpath(model_name_) # model_dir = model_root/model_name_ -> it is folder, where model script and its config file reside

# model config file
model_config_path = config['dataset_generation'].get('numerical_model_config')
# default config has same name as model and is in same folder
if model_config_path == 'default' or model_config_path == '':
  model_config_path = model_dir.joinpath(model_name_+'.ini') # model_dir/model_name_.ini 
else:
  model_config_path = model_config_path.replace('PRJ_ROOT', prj_root)
  model_config_path = Path(model_config_path)


# dataset size
N = config['dataset_generation'].getint('N') # num of dataset points
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
dataset_root_ = config['dataset_generation'].get('dataset_dir') # keep original string for dataset config
dataset_root = dataset_root_.replace('PRJ_ROOT', prj_root)
dataset_root = Path(dataset_root)

dryrun = config['dataset_generation'].getint('dryrun') # visualize a single simulation run or save full dataset

dev_ = config['dataset_generation'].get('dev') # cpu or gpu, keep original for dataset config
dev = dev_
#-------------------------------------------------------------------------------




# load model
# we want to load the package through potential subfolders
# we can pretend we are in the PRJ_ROOT, for __import__ will look for the package from there
model_path_folders = Path(model_root_.replace('PRJ_ROOT', '.')).joinpath(model_name_).parts # also add folder with same name as model

# create package structure by concatenating folders with '.'
packages_struct = model_path_folders[0]
for pkg in range(1,len(model_path_folders)):
    packages_struct += '.'+model_path_folders[pkg] 
# load 
model = __import__(packages_struct + '.' + model_name_, fromlist=['*']) # model.path.model_name_ is model script [i.e., package]

# in case of generic gpu or cuda explicitly, check if available
if dev == 'gpu' or 'cuda' in dev:
  if torch.cuda.is_available():  
    dev = torch.device('cuda')
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
  else:  
    dev = torch.device('cpu')
    print('dataset_generator: gpu not avaialable!')

print('Device:', dev)



# either generate full dataset and save it
if dryrun == 0:

  # compute name of dataset

  # count datastes in folder 
  datasets = list(Path(dataset_root).glob('*'))
  num_of_datasets = len(datasets)
  # choose new dataset index accordingly
  DATASET_INDEX = str(num_of_datasets)

  name_clash = True

  while name_clash:
    name_clash = False
    for dataset in datasets:
      # in case a dataset with same name is there
      if Path(dataset).parts[-1] == 'dataset_'+DATASET_INDEX:
        name_clash = True
        DATASET_INDEX = str(int(DATASET_INDEX)+1) # increase index

  dataset_name = 'dataset_'+DATASET_INDEX
  dataset_dir = dataset_root.joinpath(dataset_name)

  # create folder where to save dataset
  dataset_dir.mkdir(parents=True, exist_ok=True)

  #-------------------------------------------------------------------------------
  # load model and compute meta data, e.g., duration, actual size...

  nn = model.load(model_name_, model_config_path, w, h)
  n_zeros = 0
  # if this is a grid model...
  if nn != None:
    # ...we ignore the requested data point num and use the one built into the model [i.e., number of grid inputs]
    if nn%B > 0:
      # if num of data poins is not an integer multiple of batch size
      N = B * (1 + nn//B) # we round it up to the closest multiple of batch size, because this is what the model will do, 
      # it will add silence for all the simulations that exceed the number of grid inputs defined in the model's ini file
      n_zeros = N-nn # we save the number of zero data points that are addeded at the end of dataset
    else:
      # otherwise, no need to round up
      N = nn
  
  time_duration = nsteps/samplerate 
  print('simulation duration: ', time_duration, 's')

  # num of chunks must be lower than total number of batches
  if ch > N//B:
    ch = (N//B)//2 # otherwise, a chunk every other batch
  if ch == 0: # always at least one chunk!
    ch = 1

  
  # it is possible that not all requested points are generated,
  # depending on ratio with requested batch size
  # this is why we calculate the actual_size of the dataset once saved
  # this is not true for grid-based models [they return have fixed numbers of points that takes batch size into account]
  n_cnt=0
  num_of_batches = N//B
  batches_per_ch = num_of_batches//ch
  ch_size = batches_per_ch * B # num of data points per chunk
  ch_cnt = 0
  rem = 0 # is there any remainder?

  # compute number of leading zeros for pretty file names
  ch_num = str(ch)
  l_zeros=len(ch_num) # number of leading zeros in file name
  # check if l_zeros needs to be lowered down
  # e.g., ch_num = 100 -> [0, 99] -> should be printed with only one leading zero:
  # 01, 02, ..., 98, 99
  cc = pow(10,l_zeros-1)
  if ch <= cc:
    l_zeros = l_zeros-1 



  #-------------------------------------------------------------------------------
  # compute full dataset and save it

  t1 = default_timer()

  # initial conditions
  #a = torch.zeros(ch_size, h, w) #VIC we will re-introduce at a certain point, to save continous excitation and other parameters, like mu and boundaries [first static then dynamic]
  # solutions
  u = torch.zeros(ch_size, h, w, nsteps+1) # +1 becase initial condition is saved at beginning of solution time series!


  for b in range(num_of_batches):
    # compute all steps in full batch
    sol, sol_t = model.run(dev, 1/samplerate, nsteps, B)
    
    # store
    u[n_cnt:(n_cnt+B),...] = sol # results

    n_cnt += B

    # save some chunk, just in case...
    if (b+1) % batches_per_ch == 0: 
      file_name = dataset_name + '_ch' + str(ch_cnt).zfill(l_zeros) + '_' + str(n_cnt) + '.mat'
      dataset_path = dataset_dir.joinpath(file_name)
      #scipy.io.savemat(dataset_path, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
      scipy.io.savemat(dataset_path, mdict={'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
      print( '\tchunk {}, {} dataset points  (up to batch {} of {})'.format(ch_cnt, ch_size, b+1, num_of_batches) )
      ch_cnt += 1
      # reset initial conditions, solutions and data point count
      u = torch.zeros(ch_size, h, w, nsteps+1) # +1 becase initial condition is repeated at beginning of solution time series!
      n_cnt = 0
    elif (b+1) == num_of_batches:
      file_name = dataset_name + '_rem_' + str(n_cnt)  + '.mat'
      dataset_path = dataset_dir.joinpath(file_name)
      scipy.io.savemat(dataset_path, mdict={'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
      print( '\tremainder, {} dataset points (up to batch {} of {})'.format(n_cnt, b+1, num_of_batches) )
      rem = 1

  t2 = default_timer()

  rem_size = rem*n_cnt
  actual_size = (ch_size * ch) + rem_size

  print(f'\nDataset {dataset_name} saved in:')
  print('\t', dataset_dir)
  print(f'total number of data points: {actual_size} (out of {N} requested)')
  if ch == 1:
    print('in a single chunk')
  else:
    print(f'split in {ch_cnt} chunks with {ch_size} datapoints each')
    if rem:
      print(f'plus remainder file with {n_cnt} datapoints')

  simulation_duration = t2-t1
  print(f'\nElapsed time:{simulation_duration} s')



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
  if n_zeros > 0:
    config.set('dataset_details', 'final_zero_N', n_zeros)

  config.add_section('dataset_generation')
  config.set('dataset_generation', 'numerical_model_dir', model_root_)
  config.set('dataset_generation', 'numerical_model', model_name_)
  # the model config file is the current config file
  # by doing so, dataset too can be re-built using this config file
  model_config_path_ = Path(dataset_root_).joinpath(dataset_name) # up to dataset folder, with PRJ_ROOT var
  dataset_config_name = dataset_name+'.ini' # same name as dataset
  model_config_path_ = model_config_path_.joinpath(dataset_config_name) # add file name
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
  config.set('dataset_generation', 'dev', dev_)
  config.set('dataset_generation', 'dryrun', 0)


  # then retrieve model and solver details from model config file
  config_model = configparser.ConfigParser(allow_no_value=True)

  try:
      with open(model_config_path) as f:
        config_model.read(model_config_path)
  except IOError:
      print(f'dataset_generator: Model config file not found --- \'{model_config_path}\'')
      quit()

  # extract relevant bits and add them to new dataset config file
  config.add_section('numerical_model_details')
  for(each_key, each_val) in config_model.items('numerical_model_details'):
      config.set('numerical_model_details', each_key, each_val)

  config.add_section('solver')
  for(each_key, each_val) in config_model.items('solver'):
      config.set('solver', each_key, each_val)
  for(each_key, each_val) in model.getSolverInfo().items():
      config.set('solver', each_key, each_val)
  
  config.add_section('numerical_model_parameters')
  for(each_key, each_val) in config_model.items('numerical_model_parameters'):
      config.set('numerical_model_parameters', each_key, each_val)

  # where to write it
  config_path = dataset_dir.joinpath(dataset_config_name) 
  # write
  with open(config_path, 'w') as configfile:
      config.write(configfile)




else:
  # or generate 1 data entry and visualize it

  model.load(model_name_, model_config_path, w, h)

  disp_rate = 1/1
  b = 1 # 1 entry batch

  sol, _ = model.run(dev, 1/samplerate, nsteps, b, True, disp_rate)