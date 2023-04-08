import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os

import numpy as np
import random as rd
import scipy.io
import h5py
import torch
import torch.nn as nn
import operator
from functools import reduce
import math



def getProjectRoot(file):
    prj_root = Path(file).absolute() # path of this script, which is in PRJ_ROOT
    prj_root = prj_root.relative_to(Path.cwd()) # path of this script, relative to the current working directory, i.e, from where the script was called 
    prj_root = str(prj_root.parent) # dir of this script, relative to working dir (i.e, PRJ_ROOT)
    return prj_root


def openConfig(config_path, caller_name):
    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str # otherwise config parser converts all entries to lower case letters

    try:
        with open(config_path) as f:
            config.read_file(f)
    except IOError:
        print(f'{caller_name}: Config file not found --- \'{config_path}\'')
        quit()
    
    return config


def getConfigParser(prj_root, caller_name):
    # parse argument to look for user ini file
    parser = argparse.ArgumentParser()
    default_config = str(Path(prj_root).joinpath('default.ini'))
    parser.add_argument('--config', type=str, default =default_config , help='path of config file')
    args = parser.parse_args()

    # get config file
    config_path = args.config
    
    return openConfig(config_path, caller_name), config_path


# for determinism in torch DataLoader https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    rd.seed(worker_seed)




#VIC this is the content of: https://github.com/zongyi-li/fourier_neural_operator/blob/master/utilities3.py

#################################################
#
# Utilities
#
#################################################
# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer_old(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                    std = self.std[...,sample_idx] + self.eps # T*batch*n
                    mean = self.mean[...,sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


#loss function with rel Lp loss and time derivatives
class LpLossDelta(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLossDelta, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel_dt(self, x, x_prev, y, y_prev, sr=44100., dt_weight=.5):
        diff_norms = torch.norm(x.reshape(x.shape[0], -1) - y.reshape(y.shape[0], -1), 
                                self.p, 
                                1)
        y_norms = torch.norm(y.reshape(y.shape[0], -1), 
                             self.p, 
                             1)

        rel_loss = diff_norms / y_norms
        
        dt = 1 / sr
        x_dt = (x - x_prev) / dt
        y_dt = (y - y_prev) / dt
        
        dt_diff_norms = torch.norm(x_dt.reshape(x.shape[0], -1) - y_dt.reshape(y.shape[0], -1), 
                                   self.p, 
                                   1)
        y_dt_norms = torch.norm(y_dt.reshape(y.shape[0], -1), 
                                self.p, 
                                1)
        
        rel_loss_dt = dt_diff_norms / y_dt_norms
        
        rel_loss_sum = (1 - dt_weight) * rel_loss + dt_weight * rel_loss_dt
        
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_loss_sum)
            else:
                return torch.sum(rel_loss_sum)

        return rel_loss_sum

    def rel_ds(self, x, y, ds=.01, ds_weight=.5):
        diff_norms = torch.norm(x.reshape(x.shape[0], -1) - y.reshape(y.shape[0], -1), 
                                self.p, 
                                1)
        y_norms = torch.norm(y.reshape(y.shape[0], -1), 
                             self.p, 
                             1)
       
        rel_loss = diff_norms / y_norms
        
        rows = x.shape[1]
        cols = x.shape[2]
        
        # Compute spatial derivatives of x
        x_pad = torch.zeros((x.shape[0], rows + 4, cols + 4, x.shape[3])).to(x.device)
        x_pad[:, 2:2+rows, 2:2+cols, :] = x[...]
        
        bh_x = (20 * x_pad[:, 2:2+rows, 2:2+cols, :] +\
               x_pad[:, 2:2+rows, :cols, :] - 8 * x_pad[:, 2:2+rows, 1:1+cols, :] -\
               8 * x_pad[:, 2:2+rows, 3:3+cols, :] + x_pad[:, 2:2+rows, 4:4+cols, :] +\
               x_pad[:, :rows, 2:2+cols, :] - 8 * x_pad[:, 1:1+rows, 2:2+cols, :] -\
               8 * x_pad[:, 3:3+rows, 2:2+cols, :] + x_pad[:, 4:4+rows, 2:2+cols, :] +\
               2 * x_pad[:, 1:1+rows, 1:1+cols, :] + 2 * x_pad[:, 3:3+rows, 1:1+cols, :] +\
               2 * x_pad[:, 1:1+rows, 3:3+cols, :] + 2 * x_pad[:, 3:3+rows, 3:3+cols, :]) / (ds**4)
        
        # Compute spatial derivatives of y
        y_pad = torch.zeros((y.shape[0], rows + 4, cols + 4, y.shape[3])).to(y.device)
        y_pad[:, 2:2+rows, 2:2+cols, :] = y[...]
        
        bh_y = (20 * y_pad[:, 2:2+rows, 2:2+cols, :] +\
               y_pad[:, 2:2+rows, :cols, :] - 8 * y_pad[:, 2:2+rows, 1:1+cols, :] -\
               8 * y_pad[:, 2:2+rows, 3:3+cols, :] + y_pad[:, 2:2+rows, 4:4+cols, :] +\
               y_pad[:, :rows, 2:2+cols, :] - 8 * y_pad[:, 1:1+rows, 2:2+cols, :] -\
               8 * y_pad[:, 3:3+rows, 2:2+cols, :] + y_pad[:, 4:4+rows, 2:2+cols, :] +\
               2 * y_pad[:, 1:1+rows, 1:1+cols, :] + 2 * y_pad[:, 3:3+rows, 1:1+cols, :] +\
               2 * y_pad[:, 1:1+rows, 3:3+cols, :] + 2 * y_pad[:, 3:3+rows, 3:3+cols, :]) / (ds**4)
        
        ds_diff_norms = torch.norm(bh_x.reshape(x.shape[0], -1) - bh_y.reshape(y.shape[0], -1), 
                                   self.p, 
                                   1)
        ds_y_norms = torch.norm(bh_y.reshape(y.shape[0], -1), 
                                self.p, 
                                1)
        
        rel_loss_ds = ds_diff_norms / ds_y_norms
        
        rel_loss_sum = (1 - ds_weight) * rel_loss + ds_weight * rel_loss_ds
        
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_loss_sum)
            else:
                return torch.sum(rel_loss_sum)

        return rel_loss_sum
    
    def rel_dtds(self, x, x_prev, y, y_prev, ds=.01, sr=44100., dt_weight=.4, ds_weight=.3):
        diff_norms = torch.norm(x.reshape(x.shape[0], -1) - y.reshape(y.shape[0], -1), 
                                self.p, 
                                1)
        y_norms = torch.norm(y.reshape(y.shape[0], -1), 
                             self.p, 
                             1)
       
        rel_loss = diff_norms / y_norms
        
        rows = x.shape[1]
        cols = x.shape[2]
        
        # Compute spatial derivatives of x
        x_pad = torch.zeros((x.shape[0], rows + 4, cols + 4, x.shape[3])).to(x.device)
        x_pad[:, 2:2+rows, 2:2+cols, :] = x[...]
        
        bh_x = (20 * x_pad[:, 2:2+rows, 2:2+cols, :] +\
               x_pad[:, 2:2+rows, :cols, :] - 8 * x_pad[:, 2:2+rows, 1:1+cols, :] -\
               8 * x_pad[:, 2:2+rows, 3:3+cols, :] + x_pad[:, 2:2+rows, 4:4+cols, :] +\
               x_pad[:, :rows, 2:2+cols, :] - 8 * x_pad[:, 1:1+rows, 2:2+cols, :] -\
               8 * x_pad[:, 3:3+rows, 2:2+cols, :] + x_pad[:, 4:4+rows, 2:2+cols, :] +\
               2 * x_pad[:, 1:1+rows, 1:1+cols, :] + 2 * x_pad[:, 3:3+rows, 1:1+cols, :] +\
               2 * x_pad[:, 1:1+rows, 3:3+cols, :] + 2 * x_pad[:, 3:3+rows, 3:3+cols, :]) / (ds**4)
        
        # Compute spatial derivatives of y
        y_pad = torch.zeros((y.shape[0], rows + 4, cols + 4, y.shape[3])).to(y.device)
        y_pad[:, 2:2+rows, 2:2+cols, :] = y[...]
        
        bh_y = (20 * y_pad[:, 2:2+rows, 2:2+cols, :] +\
               y_pad[:, 2:2+rows, :cols, :] - 8 * y_pad[:, 2:2+rows, 1:1+cols, :] -\
               8 * y_pad[:, 2:2+rows, 3:3+cols, :] + y_pad[:, 2:2+rows, 4:4+cols, :] +\
               y_pad[:, :rows, 2:2+cols, :] - 8 * y_pad[:, 1:1+rows, 2:2+cols, :] -\
               8 * y_pad[:, 3:3+rows, 2:2+cols, :] + y_pad[:, 4:4+rows, 2:2+cols, :] +\
               2 * y_pad[:, 1:1+rows, 1:1+cols, :] + 2 * y_pad[:, 3:3+rows, 1:1+cols, :] +\
               2 * y_pad[:, 1:1+rows, 3:3+cols, :] + 2 * y_pad[:, 3:3+rows, 3:3+cols, :]) / (ds**4)
        
        ds_diff_norms = torch.norm(bh_x.reshape(x.shape[0], -1) - bh_y.reshape(y.shape[0], -1), 
                                   self.p, 
                                   1)
        ds_y_norms = torch.norm(bh_y.reshape(y.shape[0], -1), 
                                self.p, 
                                1)
        
        rel_loss_ds = ds_diff_norms / ds_y_norms
        
        dt = 1 / sr
        x_dt = (x - x_prev) / dt
        y_dt = (y - y_prev) / dt
        
        dt_diff_norms = torch.norm(x_dt.reshape(x.shape[0], -1) - y_dt.reshape(y.shape[0], -1), 
                                   self.p, 
                                   1)
        y_dt_norms = torch.norm(y_dt.reshape(y.shape[0], -1), 
                                self.p, 
                                1)
        
        rel_loss_dt = dt_diff_norms / y_dt_norms
        
        rel_loss_sum = (1 - ds_weight -dt_weight) * rel_loss +\
                       ds_weight * rel_loss_ds +\
                       dt_weight * rel_loss_dt
        
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_loss_sum)
            else:
                return torch.sum(rel_loss_sum)

        return rel_loss_sum
    
    def __call__(self, x, x_prev, y, y_prev):
        return self.rel_dtds(x, x_prev, y, y_prev)



# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average
        if a == None:
            a = [1,] * k
        self.a = a
    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms
    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)
        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)
        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])
        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)
        return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c
