from neuralacoustics.utils import MatReader 
from pathlib import Path
import torch

#VIC leave this as it is

def dataset_loader(dataset_name, dataset_path, n, win, stride=1, win_lim=-1) :
  # get N, T, w and h from file name
  datadetails = dataset_name.split("_")
  N = datadetails[2][1:]
  N = int(N) # num of dataset entries
  T = datadetails[3][1:]
  T = int(T) # timesteps of each dataset entry
  w = datadetails[4][1:]
  w = int(w)
  h = int(w)
  # all the other parameters are dataset specific!

  # to grab only a subset of the timesteps in each data entry
  if win_lim != -1 and win_lim < T:
    T = win_lim

  # check that window size is smaller than number of timesteps per each data entry
  assert (T >= win)  

  # each entry in the dataset is now split in several trainig points, as big as T_in+T_out
  #p_num = T-(win-1) # number of points per each dataset entry  
  p_num = int( (T-win)/stride ) +1 # number of points per each dataset entry  
  p_tot = N * p_num # all training points in dataset
  print('Availble points in dataset: ', p_tot)
  print('Points requested: ', n)
  assert (p_tot >= n)  

  # count number of checkpoints, their size and check for remainder file
  dataset_full_path = Path(dataset_path).joinpath(dataset_name + '/')

  # Let's avoid os, and use Pathlib instead
  files = dataset_full_path.iterdir()
  files.pop(0) # ignore config file #VIC needs to be tested
  cp = len(files)
  rem = 0

  for name in files :
    splitname = name.split("_")
    if splitname[-2] == 'rem' :
      rem = 1
      cp = cp-1
      break

  files = sorted(files) # order checkpoint files
  cp_size = files[0].split("_")[-1].split(".")[0] # read number of dataset entries in each checkpoint
  cp_size = int(cp_size)
  #cp_size = N//cp # number of dataset entries in each checkpoint
  rem_size = 0 # number of dataset entries in remainder file [if any]
  if rem > 0 :
    rem_size = N - (cp*cp_size)

  #print(N, T, w, h, cp, cp_size, rem, rem_size)

  # prepare tensor where to load requested data points
  u = torch.zeros(n, h, w, win)
  #print(u.shape)

  # actual sizes with moving window

  cp_size_p = cp_size * p_num # number of points per each check point
  rem_size_p = rem_size * p_num # number of points in remainder

  # let's load

  # check how many files we need to cover n points
  full_files = n//(cp_size_p)
  extra_datapoints = n%cp_size_p

  extra_file_needed = extra_datapoints>0

  print('Retrieved over', full_files, 'full files,', cp_size_p, 'points each')

  # check that all numbers are fine
  assert (full_files+extra_file_needed <= cp+rem)

  
  #print(files)
  

  # first load from files we will read completely 
  cnt = 0
  for f in range(0,full_files) :
    dataloader = MatReader(dataset_full_path+files[f])
    uu = dataloader.read_field('u')
    #print(f, files[f])
    # unroll all entries with moving window
    for e in range(0, cp_size) :
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
    print('Plus', extra_datapoints, 'points from further file')
    extra_entries = (extra_datapoints+0.5)//p_num # ceiling to be sure to have enough entries to unroll
    dataloader = MatReader(dataset_full_path+files[full_files])
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