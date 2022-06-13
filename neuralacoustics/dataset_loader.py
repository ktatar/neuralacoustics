import torch
import configparser
from pathlib import Path
from neuralacoustics.utils import openConfig
from neuralacoustics.utils import MatReader 



def loadDataset(dataset_name, dataset_root, n, win, stride=0, win_lim=0, start_ch=0, permute=False):
  
  print('Loading dataset:', dataset_name)

  #--------------------------------------------------------------
   # get dataset log file (as config file)
  dataset_dir = Path(dataset_root).joinpath(dataset_name) # dataset_root/dataset_name
  config_path = dataset_dir.joinpath(dataset_name+'.ini') # dataset_root/dataset_name/dataset_name.ini 

  config = openConfig(config_path, Path(__file__).name) # this is an auxiliary script, not called directly from command line
  # so __file__ is a path and we need to retrieve just the file name



  #--------------------------------------------------------------
  # read from config file
  # get N, T, w and h, ch from file name
  N = config['dataset_generation'].getint('N')
  T = config['dataset_generation'].getint('nsteps') # num of timesteps in each dataset entry
  T = T+1 # starting condition + requested steps
  w = config['dataset_generation'].getint('w')
  h = config['dataset_generation'].getint('h')
  ch = config['dataset_generation'].getint('chunks') # number of chunks (files)
  
  ch_size = config['dataset_details'].getint('chunk_size') # number of entries in each chunk (files)
  rem_size = config['dataset_details'].getint('remainder_size') # number of entries in reminder file (if any)
  rem = int(rem_size > 0) # is there reminder file?

  # if the requested start chunk file is default [<0] or above number of chunks, then start from 0
  if(start_ch <0 or start_ch>ch+rem-1): # start chunk file can be remainder file
    start_ch = 0

  
  #--------------------------------------------------------------
  # data point extraction params

  # by default, each window covers all available timesteps [full simulation time]
  if(win <= 0):
    win = T

  # by default, windows are juxtaposed
  if stride <= 0:
    stride = win

  # win_limit is used to grab only a subset of timesteps in each data entry
  # by default it is ignored
  # but if is less than number of all timesteps [full simulation time] and gte window, it is applied
  if win_lim > 0 and win_lim < T and win_lim >= win:
    T = win_lim 

  # check that window size is smaller than number of timesteps per each data entry
  assert (T >= win)  

  # each entry in the dataset is now split in several trainig points, as big as T_in+T_out
  #p_num = T-(win-1) # number of points per each dataset entry  
  p_num = int( (T-win)/stride ) +1 # number of points per each dataset entry  
  p_tot = (N-ch_size*start_ch) * p_num # all training points in dataset
  if(start_ch==0):
    print(f'\tAvailble points in dataset: {p_tot}')
  else:
    print(f'\tAvailble points in dataset (starting from chunk {start_ch}): {p_tot}')
  print(f'\tPoints requested: {n}')
  assert(p_tot >= n)  

  # count number of checkpoints, their size and check for remainder file
  files = list(dataset_dir.glob('*')) #files = dataset_dir.iterdir()
  files.remove(config_path) # ignore config file
  
  files = sorted(files) # order checkpoint files
  #print(N, T, w, h, ch, ch_size, rem, rem_size)

  # prepare tensor where to load requested data points
  u = torch.zeros(n, h, w, win)
  #print(u.shape)

  # actual sizes with moving window
  ch_size_p = ch_size * p_num # number of points per each chunk
  rem_size_p = rem_size * p_num # number of points in remainder

  
  
  
  #--------------------------------------------------------------
  # let's load

  # check how many files we need to cover n points
  full_files = n//(ch_size_p)
  extra_datapoints = n%ch_size_p

  extra_file_needed = extra_datapoints>0

  print(f'\tRetrieved over {full_files} full files, {ch_size_p} points each')
  if(start_ch>0):
    print(f'\t\tstarting from chunk file {start_ch}')

  # check that all numbers are fine
  assert (full_files+extra_file_needed <= ch+rem)
  
  #print(files)
  
  # first load from files we will read completely 
  cnt = 0
  ff = start_ch
  for f in range(full_files):
    ff += f # offset is starting chunk file
    dataloader = MatReader(files[ff])
    uu = dataloader.read_field('u')
    #print('------------------', ff, files[ff])
    # unroll all entries with moving window
    for e in range(0, ch_size):
      # window extracts p_num points from each dataset entry
      #print('e', e, 'p_num', p_num)
      for tt in range(0, p_num):
        #print('tt', tt)
        t = tt*stride
        #print('t', t)
        u[cnt:cnt+1,...] = uu[e,:,:,t:t+win]
        cnt = cnt+1

  #print(cnt, extra_datapoints)
  

  # then load any possible remainder from a further file
  if extra_datapoints>0:
    print(f'\tPlus {extra_datapoints} points from further file')
    extra_entries = (extra_datapoints+0.5)//p_num # ceiling to be sure to have enough entries to unroll
    dataloader = MatReader(files[ff])
    uu = dataloader.read_field('u')
    entry = -1
    while cnt < n:
      entry = entry+1
      for tt in range(0,p_num):
        t = tt*stride
        u[cnt:cnt+1,...] = uu[entry,:,:,t:t+win] 
        cnt = cnt+1
        if cnt >= n:
          break

  # permute dataset, along first dimension of tensor, i.e., order of time simulations on whole domain [data points]
  if(permute):
    u = u[torch.randperm(u.shape[0]),...] # this is needed when using a dataset obtained with a grid approach
    print(f'\tWith permutation of points')

  return u