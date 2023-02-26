import torch
import configparser
from pathlib import Path
from neuralacoustics.utils import openConfig
from neuralacoustics.utils import MatReader


class DatasetManager:
    """
    A class for storing information of a dataset and loading data from it.
    """

    def __init__(self, dataset_name, dataset_root, verbose=True):
        """Initialize dataset configurations."""
        if verbose:
            print('Preparing dataset:', dataset_name)

        # Get dataset directory and config file
        self.dataset_dir = Path(dataset_root).joinpath(dataset_name)
        self.config_path = self.dataset_dir.joinpath(dataset_name + '.ini')

        config = openConfig(self.config_path, Path(__file__).name)

        # Read from config file
        self.N = config['dataset_details'].getint('actual_size')
        self.T = config['numerical_model_parameters'].getint('nsteps')
        self.T += 1
        self.w = config['numerical_model_parameters'].getint('w')
        self.h = config['numerical_model_parameters'].getint('h')
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

    def checkArgs(self, start_ch, T_in, T_out, stride, win_lim):
        """
        Check the validity of data query arguments and 
        modify them if necessary.
        """
        # if either T_in or T_out are 0, or if the window is larger than # of timesteps
        # set them both to default values (half of full timesteps each)
        win = T_in + T_out
        if (T_in == 0 or T_out==0 or win > self.T):
            T_in = self.T//2
            T_out = self.T-T_in
            win = T_in + T_out

        # Start from 0 if the requested start chunk file is out of bound
        if (start_ch < 0 or start_ch > self.ch + self.rem - 1):
            start_ch = 0
        # Windows  are juxtaposed by default
        if (stride <= 0):
            stride = win
        
        # if window limit out of range, clip it to all timesteps.
        if win_lim <= 0 or win_lim > self.T:
            win_lim = self.T

        # if win_limit is too low, increase it to window size
        if win_lim < win:
            win_lim = win

        # Check that window size is smaller than number of timesteps per each data entry
        assert (self.T >= win)

        return start_ch, win, stride, T_in, T_out, win_lim

    def checkDatapointNum(self, n, p_num, start_ch):
        """Check whether there are enought datapoints."""
        p_total = (self.N - self.ch_size * start_ch) * \
            p_num  # number of points in the whole dataset
        print(
            f'\tAvailable points in the dataset (starting from chunk {start_ch}): {p_total}')
        print(f'\tPoints requested: {n}')
        assert(p_total >= n)

    def loadData(self, n, T_in, T_out, stride=0, win_lim=0, start_ch=0, permute=False):
        """Load a subsection of dataset for training."""
        # Check and modify arguments
        start_ch, win, stride, T_in, T_out, win_lim = self.checkArgs(start_ch, T_in, T_out, stride, win_lim)

        # number of points per each dataset entry
        p_num = int((win_lim - win) / stride) + 1
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
            excite = dataloader.read_field('excite')
            sol = dataloader.read_field('sol')

            # Unroll all entries with moving window
            for e in range(self.ch_size):
                for tt in range(p_num):
                    t = tt * stride
                    u[cnt, :, :, 0:T_in] = sol[e, :, :, t:t+T_in] + excite[e, :, :, t+1:t+T_in+1]
                    u[cnt, :, :, T_in:T_in+T_out] = sol[e, :, :, t+T_in:t+T_in+T_out]
                    cnt += 1

        # Load the remaining file
        if (extra_datapoints > 0):
            print(f'\tPlus {extra_datapoints} points from one other file')
            dataloader = MatReader(self.files[start_ch + full_files])
            excite = dataloader.read_field('excite')
            sol = dataloader.read_field('sol')
            data_entry = 0
            while (cnt < n):
                for tt in range(p_num):
                    t = tt * stride
                    
                    u[cnt, :, :, 0:T_in] = sol[data_entry, :, :, t:t+T_in] + excite[data_entry, :, :, t+1:t+T_in+1]
                    u[cnt, :, :, T_in:T_in+T_out] = sol[data_entry, :, :, t+T_in:t+T_in+T_out]
                    cnt += 1
                    if (cnt >= n):
                        break

                data_entry += 1

        # Permute dataset if required
        if (permute):
            u = u[torch.randperm(u.shape[0]), ...]
            print(f'\tWith permutation of points')

        return u

    def loadDataEntry(self, n, T_in, T_out, entry):
        """Load data from one single data entry."""
        win = T_in + T_out
        # Check validity of entry index
        if entry >= self.ch * self.ch_size + self.rem_size or entry < 0:
            raise AssertionError("Invalid entry index")

        # Set n as the maximum timesteps if specified as -1
        if n == -1:
            n = self.T + 1 - win

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
        excite = dataloader.read_field('excite')
        sol = dataloader.read_field('sol')
        for tt in range(n):
            u[cnt, :, :, 0:T_in] = sol[entry_in_file, :, :, tt:tt+T_in] + excite[entry_in_file, :, :, tt+1:tt+T_in+1]
            u[cnt, :, :, T_in:T_in+T_out] = sol[entry_in_file, :, :, tt+T_in:tt+T_in+T_out]

            cnt += 1

        return u
