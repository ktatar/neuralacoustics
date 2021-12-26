import torch
import configparser
from pathlib import Path
from neuralacoustics.utils import MatReader 



def loadDataset(dataset_name, dataset_path, n, win, stride=0, win_lim=0) :
  
  print('Loading dataset:', dataset_name)

  
  #--------------------------------------------------------------
   # get dataset log file (as config file)
  config = configparser.ConfigParser(allow_no_value=True)
  dataset_full_path = Path(dataset_path).joinpath(dataset_name)
  config_path = dataset_full_path.joinpath(dataset_name+'.ini') # dataset_path/dataset_name/dataset_name.ini 

  try:
    with open(config_path) as f:
        config.read_file(f)
  except IOError:
      print('dataset_loader: Config gile not found --- \'{}\''.format(config_path))
      quit()



  #--------------------------------------------------------------
  # read from config file
  # get N, T, w and h, ch from file name
  N = config['dataset_generation'].getint('N')
  T = config['dataset_generation'].getint('nsteps') # num of timesteps in each dataste entry
  w = config['dataset_generation'].getint('w')
  h = config['dataset_generation'].getint('h')
  ch = config['dataset_generation'].getint('chunks') # number of chunks (files)

  ch_size = config['dataset_details'].getint('chunk_size') # number of entries in each chunk (files)
  rem_size = config['dataset_details'].getint('remainder_size') # number of entries in reminder file (if any)
  rem = int(rem_size > 0) # is there reminder file?


  
  #--------------------------------------------------------------
  # data point extraction params

  # by default, windows are juxtaposed
  if stride <= 0 :
    stride = win

  # to grab only a subset of timesteps in each data entry
  if win_lim > 0 and win_lim < T:
    T = win_lim


  # check that window size is smaller than number of timesteps per each data entry
  assert (T >= win)  

  # each entry in the dataset is now split in several trainig points, as big as T_in+T_out
  #p_num = T-(win-1) # number of points per each dataset entry  
  p_num = int( (T-win)/stride ) +1 # number of points per each dataset entry  
  p_tot = N * p_num # all training points in dataset
  print('\tAvailble points in dataset:', p_tot)
  print('\tPoints requested:', n)
  assert (p_tot >= n)  

  # count number of checkpoints, their size and check for remainder file
  files = list(dataset_full_path.glob('*')) #files = dataset_full_path.iterdir()
  files.remove(config_path) #files.pop(0) # ignore config file #VIC needs to be tested
  
  files = sorted(files) # order checkpoint files
  #print(N, T, w, h, ch, ch_size, rem, rem_size)

  # prepare tensor where to load requested data points
  u = torch.zeros(n, h, w, win)
  #print(u.shape)

  # actual sizes with moving window
  ch_size_p = ch_size * p_num # number of points per each check point
  rem_size_p = rem_size * p_num # number of points in remainder

  
  
  
  #--------------------------------------------------------------
  # let's load

  # check how many files we need to cover n points
  full_files = n//(ch_size_p)
  extra_datapoints = n%ch_size_p

  extra_file_needed = extra_datapoints>0

  print('\tRetrieved over', full_files, 'full files,', ch_size_p, 'points each')

  # check that all numbers are fine
  assert (full_files+extra_file_needed <= ch+rem)
  
  #print(files)
  
  # first load from files we will read completely 
  cnt = 0
  for f in range(0,full_files) :
    dataloader = MatReader(files[f])
    uu = dataloader.read_field('u')
    #print(f, files[f])
    # unroll all entries with moving window
    for e in range(0, ch_size) :
      # window extracts p_num points from each dataset entry
      #print('e', e, 'p_num', p_num)
      for tt in range(0, p_num) :
        #print('tt', tt)
        t = tt*stride
        #print('t', t)
        u[cnt:cnt+1,...] = uu[e,:,:,t:t+win]
        cnt = cnt+1

  #print(cnt, extra_datapoints)
  

  # then load any possible remainder from a further file
  if extra_datapoints>0 :
    print('\tPlus', extra_datapoints, 'points from further file')
    extra_entries = (extra_datapoints+0.5)//p_num # ceiling to be sure to have enough entries to unroll
    dataloader = MatReader(files[full_files])
    uu = dataloader.read_field('u')
    entry = -1
    while cnt < n :
      entry = entry+1
      for tt in range(0,p_num) :
        t = tt*stride
        u[cnt:cnt+1,...] = uu[entry,:,:,t:t+win] 
        cnt = cnt+1
        if cnt >= n :
          break

  return u