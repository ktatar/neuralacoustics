import torch
import configparser
from pathlib import Path
from neuralacoustics.utils import openConfig
from neuralacoustics.utils import MatReader


class DatasetManager:
    """
    A class for storing information of a dataset and loading data from it.
    """

    def __init__(self, dataset_name, dataset_root):
        """Initialize dataset configurations."""
        print('Preparing dataset:', dataset_name)

        # Get dataset directory and config file
        self.dataset_dir = Path(dataset_root).joinpath(dataset_name)
        self.config_path = self.dataset_dir.joinpath(dataset_name + '.ini')

        config = openConfig(self.config_path, Path(__file__).name)

        # Read from config file
        self.N = config['dataset_generation'].getint('N')
        self.T = config['dataset_generation'].getint('nsteps')
        self.T += 1
        self.w = config['dataset_generation'].getint('w')
        self.h = config['dataset_generation'].getint('h')
        self.ch = config['dataset_generation'].getint(
            'chunks')  # number of chunks
        self.ch_size = config['dataset_details'].getint(
            'chunk_size')  # number of entries in a chunk
        self.rem_size = config['dataset_details'].getint(
            'remainder_size')  # number of entries in reminder file (if any)
        self.rem = int(self.rem_size > 0)

        # Store data filenames
        self.files = list(self.dataset_dir.glob('*'))
        self.files.remove(self.config_path)  # ignore config file
        self.files = sorted(self.files)

    def checkArgs(self, start_ch, win, stride, win_lim):
        """
        Check the validity of data query arguments and 
        modify them if necessary.
        """
        # Start from 0 if the requested start chunk file is out of bound
        if (start_ch < 0 or start_ch > self.ch + self.rem - 1):
            start_ch = 0
        # Each window covers all available timesteps by default
        if (win <= 0):
            win = self.T
        # Windows are juxtaposed by default
        if (stride <= 0):
            stride = win
        # Modify total timestep if window limit is applied
        if win_lim > 0 and win_lim < self.T and win_lim >= win:
            self.T = win_lim

        # Check that window size is smaller than number of timesteps per each data entry
        assert (self.T >= win)

        return start_ch, win, stride

    def checkDatapointNum(self, n, p_num, start_ch):
        """Check whether there are enought datapoints."""
        p_total = (self.N - self.ch_size * start_ch) * \
            p_num  # number of points in the whole dataset
        print(
            f'\tAvailable points in the dataset (starting from chunk {start_ch}): {p_total}')
        print(f'\tPoints requested: {n}')
        assert(p_total >= n)

    def loadData(self, n, win, stride=0, win_lim=0, start_ch=0, permute=False):
        """Load a subsection of dataset for training."""
        # Check and modify arguments
        start_ch, win, stride = self.checkArgs(start_ch, win, stride, win_lim)

        # number of points per each dataset entry
        p_num = int((self.T - win) / stride) + 1
        # Check whether there's enough datapoints
        self.checkDatapointNum(n, p_num, start_ch)

        ch_size_p = self.ch_size * p_num  # number of points in one full file
        full_files = n // ch_size_p
        extra_datapoints = n % ch_size_p

        print(
            f'\tRetrieving from {full_files} full files, {ch_size_p} points each')
        print(f'\t\tStarting from chunk file {start_ch}')

        # Start loading data
        # Prepare tensor where to load requested data points
        u = torch.zeros(n, self.h, self.w, win)

        # Load from the files to be completely read
        cnt = 0
        for f in range(full_files):
            dataloader = MatReader(self.files[f + start_ch])
            uu = dataloader.read_field('u')

            # Unroll all entries with moving window
            for e in range(self.ch_size):
                for tt in range(p_num):
                    t = tt * stride
                    u[cnt, ...] = uu[e, :, :, t:t+win]
                    cnt += 1

        # Load the remaining file
        if (extra_datapoints > 0):
            print(f'\tPlus {extra_datapoints} points from one other file')
            dataloader = MatReader(self.files[start_ch + full_files])
            uu = dataloader.read_field('u')
            data_entry = 0
            while (cnt < n):
                for tt in range(p_num):
                    t = tt * stride
                    u[cnt, ...] = uu[data_entry, :, :, t:t+win]
                    cnt += 1
                    if (cnt >= n):
                        break

                data_entry += 1

        # Permute dataset if required
        if (permute):
            u = u[torch.randperm(u.shape[0]), ...]
            print(f'\tWith permutation of points')

        return u

    def loadDataEntry(self, n, win, entry):
        """Load data from one single data entry."""
        # Check validity of entry index
        if entry >= self.ch * self.ch_size + self.rem_size or entry < 0:
            raise AssertionError("Invalid entry index")

        # Check whether request is out of bount
        assert (n + win <= self.T + 1)

        # Prepare tensor where to load requested data points
        u = torch.zeros(n, self.h, self.w, win)

        # Find file index and entry index in the target file
        cnt = 0
        file_index = entry // self.ch_size
        entry_in_file = entry % self.ch_size

        # Load the target entry
        dataloader = MatReader(self.files[file_index])
        uu = dataloader.read_field('u')
        for tt in range(n):
            u[cnt, ...] = uu[entry_in_file, :, :, tt:tt+win]
            cnt += 1

        return u


# def loadDataset(dataset_name, dataset_root, n, win, stride=0, win_lim=0, start_ch=0, entry=-1, permute=False):
#   """Load dataset from .mat file and return as PyTorch tensors."""
#   print('Loading dataset:', dataset_name)

#   #--------------------------------------------------------------
#    # get dataset log file (as config file)
#   dataset_dir = Path(dataset_root).joinpath(dataset_name) # dataset_root/dataset_name
#   config_path = dataset_dir.joinpath(dataset_name+'.ini') # dataset_root/dataset_name/dataset_name.ini

#   config = openConfig(config_path, Path(__file__).name) # this is an auxiliary script, not called directly from command line
#   # so __file__ is a path and we need to retrieve just the file name


#   #--------------------------------------------------------------
#   # read from config file
#   # get N, T, w and h, ch from file name
#   N = config['dataset_generation'].getint('N')
#   T = config['dataset_generation'].getint('nsteps') # num of timesteps in each dataset entry
#   T = T+1 # starting condition + requested steps
#   w = config['dataset_generation'].getint('w')
#   h = config['dataset_generation'].getint('h')
#   ch = config['dataset_generation'].getint('chunks') # number of chunks (files)

#   ch_size = config['dataset_details'].getint('chunk_size') # number of entries in each chunk (files)
#   rem_size = config['dataset_details'].getint('remainder_size') # number of entries in reminder file (if any)
#   rem = int(rem_size > 0) # is there reminder file?

#   # if the requested start chunk file is default [<0] or above number of chunks, then start from 0
#   if(start_ch <0 or start_ch>ch+rem-1): # start chunk file can be remainder file
#     start_ch = 0

#   # Check validity of entry index
#   if entry >= ch * ch_size + rem_size:
#     raise AssertionError("Entry index larger than dataset size")


#   #--------------------------------------------------------------
#   # data point extraction params

#   # by default, each window covers all available timesteps [full simulation time]
#   if(win <= 0):
#     win = T

#   # by default, windows are juxtaposed
#   if stride <= 0:
#     stride = win

#   # win_limit is used to grab only a subset of timesteps in each data entry
#   # by default it is ignored
#   # but if is less than number of all timesteps [full simulation time] and gte window, it is applied
#   if win_lim > 0 and win_lim < T and win_lim >= win:
#     T = win_lim

#   # check that window size is smaller than number of timesteps per each data entry
#   assert (T >= win)

#   # each entry in the dataset is now split in several trainig points, as big as T_in+T_out
#   #p_num = T-(win-1) # number of points per each dataset entry
#   p_num = int( (T-win)/stride ) +1 # number of points per each dataset entry
#   p_tot = (N-ch_size*start_ch) * p_num # all training points in dataset
#   if(start_ch==0):
#     print(f'\tAvailble points in dataset: {p_tot}')
#   else:
#     print(f'\tAvailble points in dataset (starting from chunk {start_ch}): {p_tot}')
#   print(f'\tPoints requested: {n}')
#   assert(p_tot >= n)

#   # count number of checkpoints, their size and check for remainder file
#   files = list(dataset_dir.glob('*')) #files = dataset_dir.iterdir()
#   files.remove(config_path) # ignore config file

#   files = sorted(files) # order checkpoint files
#   #print(N, T, w, h, ch, ch_size, rem, rem_size)

#   # prepare tensor where to load requested data points
#   u = torch.zeros(n, h, w, win)
#   #print(u.shape)

#   # actual sizes with moving window
#   ch_size_p = ch_size * p_num # number of points per each chunk
#   rem_size_p = rem_size * p_num # number of points in remainder


#   #--------------------------------------------------------------
#   # let's load

#   # Load only the target entry if entry is specified
#   if entry >= 0:
#     cnt = 0
#     file_index = entry // ch_size
#     entry_in_file = entry % ch_size

#     dataloader = MatReader(files[file_index])
#     uu = dataloader.read_field('u')
#     for tt in range(0, p_num):
#       t = tt*stride

#       u[cnt:cnt+1,...] = uu[entry_in_file,:,:,t:t+win]
#       cnt = cnt+1

#     return u

#   # check how many files we need to cover n points
#   full_files = n//(ch_size_p)
#   extra_datapoints = n%ch_size_p

#   extra_file_needed = extra_datapoints>0

#   print(f'\tRetrieved over {full_files} full files, {ch_size_p} points each')
#   if(start_ch>0):
#     print(f'\t\tstarting from chunk file {start_ch}')

#   # check that all numbers are fine
#   assert (full_files+extra_file_needed <= ch+rem)

#   #print(files)

#   # first load from files we will read completely
#   cnt = 0
#   ff = start_ch # offset is starting chunk file
#   for f in range(full_files):
#     dataloader = MatReader(files[ff])
#     ff += 1
#     uu = dataloader.read_field('u')
#     #print('------------------', ff, files[ff])
#     # unroll all entries with moving window
#     for e in range(0, ch_size):
#       # window extracts p_num points from each dataset entry
#       #print('e', e, 'p_num', p_num)
#       for tt in range(0, p_num):
#         #print('tt', tt)
#         t = tt*stride
#         #print('t', t)
#         u[cnt:cnt+1,...] = uu[e,:,:,t:t+win]
#         cnt = cnt+1

#   #print(cnt, extra_datapoints)


#   # then load any possible remainder from a further file
#   if extra_datapoints>0:
#     print(f'\tPlus {extra_datapoints} points from further file')
#     extra_entries = (extra_datapoints+0.5)//p_num # ceiling to be sure to have enough entries to unroll
#     dataloader = MatReader(files[ff]) # ff is file after the last full file loaded
#     uu = dataloader.read_field('u')
#     entry = -1
#     while cnt < n:
#       entry = entry+1
#       for tt in range(0,p_num):
#         t = tt*stride
#         u[cnt:cnt+1,...] = uu[entry,:,:,t:t+win]
#         cnt = cnt+1
#         if cnt >= n:
#           break

#   # permute dataset, along first dimension of tensor, i.e., order of time simulations on whole domain [data points]
#   if(permute):
#     u = u[torch.randperm(u.shape[0]),...] # this is needed when using a dataset obtained with a grid approach
#     print(f'\tWith permutation of points')

#   return u
