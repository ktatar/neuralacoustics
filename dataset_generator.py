import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os
import torch
from timeit import default_timer # to measure processing time
import scipy.io # to save dataset


#-------------------------------------------------------------------------------
# simulation parameters

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='./default.ini' , help='path to the config file')
args = parser.parse_args()

# Get config file
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
try:
  config.read(config_path)
except FileNotFoundError:
  print('dataset_generator: Config File not found in {}'.format(config_path))
  sys.exit()


# read params from config file

# model
model_root = Path(config['dataset_generation'].get('numerical_model_path'))
model_name = config['dataset_generation'].get('numerical_model')

model_path = model_root.joinpath(model_name) # model_path = model_root/model_name -> it is folder, where model script and its config file reside

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

# dataset path
dataset_root = Path(config['dataset_generation'].get('dataset_path'))

dryrun = config['dataset_generation'].getint('dryrun') # visualize a single simulation run or save full dataset

#-------------------------------------------------------------------------------




# load model
model = __import__(str(model_root) + '.' + model_name + '.' + model_name, fromlist=['*']) # models_root.model_name.model_name is model script


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"

print('device:', dev)




if dryrun == 0 :
  # either generate Full Dataset and Save it

  # count datastes in folder 
  num_of_datasets = len(list(Path(dataset_root).glob('*')))
  # choose new dataset index accordingly
  DATASET_INDEX = str(num_of_datasets+1)

  time_duration = nsteps/samplerate 
  print('simulation duration: ', time_duration, 's')

  # num of chunks must be lower than total number of batches
  if ch > N//B :
    ch = (N//B)//2 # a chunk every other batch
  if ch == 0 :
    ch = 1

  
  dataset_name = 'dataset_'+DATASET_INDEX
  dataset_path = dataset_root.joinpath(dataset_name)

  print('Dataset path: ', dataset_path)


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
  dataset_path.mkdir(parents=True, exist_ok=True)


  # compute number of leading zeros for pretty file names
  ch_num = str(ch)
  l_zeros=len(ch_num) # number of leading zeros in file name
  # check if l_zeros needs to be lowered down
  # e.g., ch_num = 100 -> [0, 99] -> should be printed with only one leading zero:
  # 01, 02, ..., 98, 99
  cc = pow(10,l_zeros-1)
  if ch <= cc :
    l_zeros = l_zeros-1 


  t1 = default_timer()

  # initial conditions
  a = torch.zeros(ch_size, h, w)
  # solutions
  u = torch.zeros(ch_size, h, w, nsteps)


  for b in range(num_of_batches):
    # compute all steps in full batch
    sol, sol_t, p0 = model.run(dev, 1/samplerate, nsteps, B, w, h, model_path, model_name)
    
    # store
    a[n_cnt:(n_cnt+B),...] = p0 # initial condition
    u[n_cnt:(n_cnt+B),...] = sol # results

    n_cnt += B

    # save some chunk, just in case...
    if (b+1) % batches_per_ch == 0 : 
      file_name = dataset_name + '_ch' + str(ch_cnt).zfill(l_zeros) + '_' + str(n_cnt) + '.mat'
      file_name = dataset_path.joinpath(file_name)
      scipy.io.savemat(file_name, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
      print( '\tchunk {}, {} dataset points  (up to batch {} of {})'.format(ch_cnt, ch_size, b+1, num_of_batches) )
      ch_cnt += 1
      # reset initial conditions, solutions and data point count
      a = torch.zeros(ch_size, h, w)
      u = torch.zeros(ch_size, h, w, nsteps)
      n_cnt = 0
    elif (b+1) == num_of_batches :
      file_name = dataset_name + '_rem_' + str(n_cnt)  + '.mat'
      file_name = dataset_path.joinpath(file_name)
      scipy.io.savemat(file_name, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
      print( '\tremainder, {} dataset points (up to batch {} of {})'.format(n_cnt, b+1, num_of_batches) )
      rem = 1

  t2 = default_timer()

  rem_size = rem*n_cnt
  actual_size = (ch_size * ch) + rem_size

  print('\nDataset', dataset_name, 'saved in:')
  print('\t', dataset_path)
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
  config.set('dataset_details', 'N', N)
  config.set('dataset_details', 'B', B)
  config.set('dataset_details', 'actual_size', actual_size)
  config.set('dataset_details', 'w', w)
  config.set('dataset_details', 'h', h)
  config.set('dataset_details', 'samplerate', samplerate)
  config.set('dataset_details', 'nsteps', nsteps)
  config.set('dataset_details', 'simulated_time_s', time_duration)
  config.set('dataset_details', 'simulation_duration_s', simulation_duration)
  config.set('dataset_details', 'chunks', ch_cnt)
  config.set('dataset_details', 'chunk_size', ch_size)
  config.set('dataset_details', 'remainder_size', rem_size)
  config.set('dataset_details', 'numerical_model', model_name)

  # then retrieve model and solver details from model config file
  model_config = configparser.ConfigParser(allow_no_value=True)
  model_config_path = model_path.joinpath(model_name+'.ini') # model_path/model_name.ini 
  try:
    model_config.read(model_config_path)
  except FileNotFoundError:
      print('dataset_generator: Model Config File Not Found In {}'.format(model_config_path))
      sys.exit()

  # extract relevant bits and add them to new dataset config file
  config.add_section('numerical_model_details')
  config.set('numerical_model_details', 'path', './'+str(model_root))
  for (each_key, each_val) in model_config.items('details') :
      config.set('numerical_model_details', each_key, each_val)
  config.set('numerical_model_details', 'solver', model_config['solver'].get('solver_name'))

  config.add_section('numerical_model_parameters')
  for (each_key, each_val) in model_config.items('parameters') :
      config.set('numerical_model_parameters', each_key, each_val)

  config.add_section('solver_details')
  solver_path = model_config['solver'].get('solver_path')
  solver_path = solver_path.replace('../', '')
  config.set('solver_details', 'path', './'+solver_path)
  config.set('solver_details', 'description', model.getSolverDescription())

  # where to write it
  config_path = dataset_path.joinpath(dataset_name+'.log')
  # write
  with open(config_path, 'w') as configfile:
      config.write(configfile)




else :
  # or generate 1 data entry and visualize it

  disp_rate = 1/1
  b=1 # 1 entry batch

  sol, _, _ = model.run(dev, 1/samplerate, nsteps, b, w, h, model_path, model_name, True, disp_rate)
