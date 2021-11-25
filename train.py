#VIC put here all the new imports that may be needed

import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
# from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from neuralacoustics.adam import Adam # adam implementation that deals with complex tensors correctly [lacking in pytorch 1.8]

import os, sys, configparser, argparse
from pathlib import Path
from neuralacoustics.utils import LpLoss
from neuralacoustics.model import Net2d
from neuralacoustics.dataset import dataset_loader
 
# KIVANC: We should check determinism on pytorch. There is a full article about it, 
# and there is a way to force pytorch to be strictly deterministic:
# https://pytorch.org/docs/stable/notes/randomness.html
# If we go this way, we should avoid any numpy ops, and use torch ops at all times. 
torch.manual_seed(0)
np.random.seed(0)

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='./default.ini' , help='path to the config file')
args = parser.parse_args()

#Get configs
config_path = Path(args.config)
config = configparser.ConfigParser(allow_no_value=True)

try: 
  config.read(config_path)

except FileNotFoundError:
  print('Config File Not Found at {}'.format(config_path))
  sys.exit()

# simulation parameters
ntrain = config['simulation'].getint('ntrain')
ntest = config['simulation'].getint('ntest')

modes = config['network'].getint('modes')
width = config['network'].getint('width')

batch_size = config['training'].getint('batch_size')

epochs = config['training'].getint('epochs')
learning_rate = config['training'].getfloat('learning_rate')
scheduler_step = config['training'].getint('scheduler_step')
scheduler_gamma = config['training'].getfloat('scheduler_gamma')

print(epochs, learning_rate, scheduler_step, scheduler_gamma)


T_in = config['simulation'].getint('window_size')
T_out = T_in
# T_in+T_out is window size!

win_stride = config['simulation'].getint('win_stride')
win_lim = config['simulation'].getint('win_lim')

#(T_in+T_out)*200 #-1 for no limit

# dataset
dataset_name = config['dataset'].get('name')

#-------------------------------------------------------------------------------



MODEL_ID = config['dataset'].get('model_id')

dataset_path = Path(config['dataset'].get('path'))

# retrieve dataset details and check them
splitname = dataset_name.split('_')

DATASET = splitname[1]

S = splitname[4]
S = int(S[1:])

mu = splitname[5][2:]
rho = splitname[6][3:]
gamma = splitname[7][5:]

# prepare to save model
model_name = 'iwe_m'+MODEL_ID+'_'+DATASET+'_n'+str(ntrain)+'+'+str(ntest)+'_e'+str(epochs)+'_m'+str(modes)+'_w'+ str(width)+'_ti'+str(T_in)+'_to'+str(T_out)+'_ws'+str(win_stride)+'_wl'+str(win_lim)+'_s'+str(S)+'_m'+mu+'_r'+rho+'_g'+gamma

model_path = dataset_path.joinpath('models')

#VIC i think this needs only a minor update, i.e., removing the padding of the location

t1 = default_timer()

u = dataset_loader(dataset_name, dataset_path, ntrain+ntest, T_in+T_out, win_stride, win_lim)

train_a = u[:ntrain,:,:,:T_in]
train_u = u[:ntrain,:,:,T_in:T_in+T_out]

ntest_start = ntrain
test_a = u[ntest_start:ntest_start+ntest,:,:,:T_in]
test_u = u[ntest_start:ntest_start+ntest:,:,:,T_in:T_in+T_out]

#test_a = u[-ntest:,:,:,:T_in]
#test_u = u[-ntest:,:,:,T_in:T_in+T_out]


print(train_u.shape, test_u.shape)
assert (S == train_u.shape[-2])
assert (T_out == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

#VIC should be removed
# pad the location (x,y)
# gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
# gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
# gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
# gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])

# train_a = torch.cat((gridx.repeat([ntrain,1,1,1]), gridy.repeat([ntrain,1,1,1]), train_a), dim=-1)
# test_a = torch.cat((gridx.repeat([ntest,1,1,1]), gridy.repeat([ntest,1,1,1]), test_a), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1, 's')
print('train input shape:',train_a.shape, ' output shape: ', train_u.shape)    

#VIC this needs a couple of touch ups, as at the bottom of: https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d_time.py
# also, i made very minor changes to the original code, to make the logic clearer

if torch.cuda.is_available() :
  model = Net2d(modes, width, T_in).cuda()
  device  = torch.device('cuda')
  print(" Utilizing CUDA")  
else :
  model = Net2d(modes, width, T_in)
  device  = torch.device('cpu')
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())    


print(model.count_params())
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


# myloss = LpLoss(size_average=False)

#VIC these are not needed anymore
# gridx = gridx.to(device)
# gridy = gridy.to(device)



myloss = LpLoss(size_average=False)
step = 1
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

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
            xx = xx.to(device)
            yy = yy.to(device)

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
    print(ep, t2 - t1, train_l2_step / ntrain / (T_out / step), train_l2_full / ntrain, test_l2_step / ntest / (T_out / step),
          test_l2_full / ntest)
    
# add loss to name, with 4 decimals    
final_training_loss = '{:.4f}'.format(test_l2_full / ntest)
final_training_loss = final_training_loss.replace('.', '@')

model_name = model_name+'_loss'+final_training_loss   
model_full_path = model_path.joinpath(model_name)

torch.save(model, model_full_path)

# save config file #TODO save only relevant bits + results
config_path = model_path.joinpath(model_name+'_config.ini')

with open(config_path, 'w') as configfile:
  config.write(configfile)