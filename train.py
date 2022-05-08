#VIC put here all the new imports that may be needed

import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from timeit import default_timer

from neuralacoustics.adam import Adam # adam implementation that deals with complex tensors correctly [lacking in pytorch 1.8]

import os, sys, configparser
from pathlib import Path

from neuralacoustics.utils import LpLoss
from neuralacoustics.model import FNO2d
from neuralacoustics.dataset_loader import loadDataset # to load dataset
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.utils import count_params


# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)

#VIC check this
# KIVANC: We should check determinism on pytorch. There is a full article about it, 
# and there is a way to force pytorch to be strictly deterministic:
# https://pytorch.org/docs/stable/notes/randomness.html
# If we go this way, we should avoid any numpy ops, and use torch ops at all times. 
torch.manual_seed(0)
np.random.seed(0)


#-------------------------------------------------------------------------------
# training parameters

# get config file
config = getConfigParser(prj_root, __file__.replace('.py', ''))

# read params from config file

# dataset parameters
# dataset name
dataset_name = config['training'].get('dataset_name')
# dataset dir
dataset_dir = config['training'].get('dataset_dir')
dataset_dir = dataset_dir.replace('PRJ_ROOT', prj_root)

# number of data points to load, i.e., specific sub-series of time steps within data entries
n_train = config['training'].getint('n_train')
n_test = config['training'].getint('n_test')

# input and output steps
T_in = config['training'].getint('T_in') 
T_out = config['training'].getint('T_out')
# T_in+T_out is window size!

# offset between consecutive windows
win_stride = config['training'].getint('window_stride') 
# by default, windows are juxtaposed
if win_stride <= 0:
    win_stride = T_in+T_out

# maximum index of the frame (timestep) that can be retrieved from each dataset entry
win_limit = config['training'].getint('window_limit') 


# network parameters
modes = config['training'].getint('network_modes')
width = config['training'].getint('network_width')


# training parameters
batch_size = config['training'].getint('batch_size')

epochs = config['training'].getint('epochs')

learning_rate = config['training'].getfloat('learning_rate')
scheduler_step = config['training'].getint('scheduler_step')
scheduler_gamma = config['training'].getfloat('scheduler_gamma')


# misc parameters
model_root = config['training'].get('model_dir')
model_root = model_root.replace('PRJ_ROOT', prj_root)
model_root = Path(model_root)

dev = config['training'].get('dev')


print('Model and training parameters:')
print(f'\tdataset name: {dataset_name}')
print(f'\trequested training data points: {n_train}')
print(f'\trequested training test points: {n_test}')
print(f'\tinput steps: {T_in}')
print(f'\toutput steps: {T_out}')
print(f'\tmodes: {modes}')
print(f'\twidth: {width}')
print(f'\tlearning_rate: {learning_rate}')
print(f'\tscheduler_step: {scheduler_step}')
print(f'\tscheduler_gamma: {scheduler_gamma}')


#-------------------------------------------------------------------------------
# compute name of model

# count datastes in folder 
models = list(Path(model_root).glob('*'))
num_of_models = len(models)
# choose new model index accordingly
MODEL_INDEX = str(num_of_models)

name_clash = True

while name_clash:
  name_clash = False
  for model in models:
    # in case a model with same name is there
    if Path(model).parts[-1] == 'model_'+MODEL_INDEX:
      name_clash = True
      MODEL_INDEX = str(int(MODEL_INDEX)+1) # increase index

model_name = 'model_'+MODEL_INDEX
model_dir = model_root.joinpath(model_name)

  # create folder where to save model
model_dir.mkdir(parents=True, exist_ok=True)


#-------------------------------------------------------------------------------
# retrieve all data points

t1 = default_timer()

u = loadDataset(dataset_name, dataset_dir, n_train+n_test, T_in+T_out, win_stride, win_limit)
# get domain size
sh = list(u.shape)
S = sh[1] 
# we assume that all datasets have simulations spanning square domains
assert(S == sh[2])

# prepare train and and test sets 
train_a = u[:n_train,:,:,:T_in]
train_u = u[:n_train,:,:,T_in:T_in+T_out]
test_a = u[-n_test:,:,:,:T_in]
test_u = u[-n_test:,:,:,T_in:T_in+T_out]

#print(train_u.shape, test_u.shape)
assert(S == train_u.shape[-2])
assert(T_out == train_u.shape[-1])

train_a = train_a.reshape(n_train,S,S,T_in)
test_a = test_a.reshape(n_test,S,S,T_in)

# datapoints will be loaded from these
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print(f'\nPreprocessing finished, time used: {t2-t1}s')
print(f'Training input shape: {train_a.shape}, output shape: {train_u.shape}')    



#-------------------------------------------------------------------------------
# select device and create model

# in case of generic gpu or cuda explicitly, check if available
if dev == 'gpu' or 'cuda' in dev:
  if torch.cuda.is_available():  
    model = FNO2d(modes, modes, width, T_in).cuda()
    dev  = torch.device('cuda')
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
  model = FNO2d(modes, modes, width, T_in)
  dev  = torch.device('cpu')

print('Device:', dev)


print(f'Model parameters number: {count_params(model)}')
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


#-------------------------------------------------------------------------------
# train!

#VIC test
# train_features, train_labels = next(iter(train_loader))
# print(train_features.size())
# print(train_labels.size())
# t1 = default_timer()
# im = model(train_features)
# t2 = default_timer()
# print(f'\nInference finished, time used: {t2-t1}s')
# quit()

myloss = LpLoss(size_average=False)
step = 1
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(dev)
        yy = yy.to(dev)

        # model outputs 1 timestep at a time [i.e., labels], so we iterate over T_out steps to compute loss
        for t in range(0, T_out, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(dev)
            yy = yy.to(dev)

            for t in range(0, T_out, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / n_train / (T_out / step), train_l2_full / n_train, test_l2_step / n_test / (T_out / step),
          test_l2_full / n_test)
    
# add loss to name, with 4 decimals    
# final_training_loss = '{:.4f}'.format(test_l2_full / n_test)
# final_training_loss = final_training_loss.replace('.', '@')

# model_name = model_name+'_loss'+final_training_loss   
# model_full_path = model_path.joinpath(model_name)

# save model to folder
model_path = model_dir.joinpath(model_name)
torch.save(model, model_path)

# # save config file #TODO save only relevant bits + results
# config_path = model_path.joinpath(model_name+'_config.ini')

# with open(config_path, 'w') as configfile:
#   config.write(configfile)