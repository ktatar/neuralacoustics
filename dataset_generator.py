# Wave equation, irreducible form. 

# Generation of 2D dataset 0:


# Rectangular domain
# Only air cells, except for outer frame of wall cells [boundaries]
# acoustics parameters, including those of boundaries are shared among all cells
# Initial conditions: gaussian noise, non normalized, accross full domain
# Includes also impulse and sinusoidal continuous excitation for testing purposes
# Editable parameters:
# > N = num of dataset points
# > nsteps = num of timesteps per point
# > s = size of domain (width = s and height = s )
# > mu = attenuation coefficient for all air cells
# > rho = propagation coefficient [material and/or scale] for all air cells
# > gamma = behaviors of boundaries [from clamped to free edges]
# > B = batch size
# > cp = number of checkpoints [i.e., number of files in which the dataset will be split]

import numpy as np
import torch
import time 
import math
import scipy.io # to save dataset
import configparser, argparse # to read config from ini file
from timeit import default_timer # to measure processing time
from pathlib import Path # to properly handle paths and folders on every os
import matplotlib.pyplot as plt # to plot dryrun
import matplotlib.colors as mcolors # to potentially use different color colormaps


# to visualize result
color_halfrange = 1
maxAmp = 20
log_min = 10.0

def plotDomain(data):
  log_data = torch.abs(data / maxAmp)
  log_data = (torch.log(log_data + 1e-8) + log_min)/log_min
  log_data = torch.clamp(log_data, 0, 1)*torch.sign(data)

  img = data.cpu().detach().numpy()
  plt.imshow(img, vmin=-color_halfrange, vmax=color_halfrange)
  #plt.imshow(img, vmin=-color_halfrange, vmax=color_halfrange, cmap=hdh_cmap) # custom colormap
  # all this non-sense is needed to have a proper non-blocking plot
  plt.ion()
  plt.show()
  plt.pause(0.001)



# solver
def irreducible_wave_equation(dev, b, p0, w, h, dt, nsteps, mu, rho, gamma, disp=False, dispRate=1):#, exciteOn=0, freq=440, excitation_x=1, excitation_y=1) :
  if disp :
    if dispRate > 1 :
      dispRate = 1
    disp_delta = 1/disp_rate
  
  # data structures

  # propagation 
  prop = torch.zeros([b, h, w, 2], device=dev) # last dimension contains: mu and rho
  prop[:,:,:,0] = mu
  prop[:,:,:,1] = rho

  # boundaries
  bound = torch.zeros([b, h, w, 2], device=dev) # last dimension contains: is wall [0-no, 1-yes] and gamma [0-clamped edge, 1-free edge] -> wall indices are opposite than Hyper Drumhead case
  # create edges all around domain
  bound[:,:,0,0] = 1
  bound[:,:,w-1,0] = 1
  bound[:,0,:,0] = 1
  bound[:, h-1,:,0] = 1
  # assign gamma to edges
  #bound[:,:,:,1] = prop[:,:,:,1]*gamma
  bound[:,:,:,1] = gamma

  # pressure
  p = torch.zeros([b, h, w, 3], device=dev) # last dimension contains: p prev, p now, p next
  # excitation
  #excite = torch.FloatTensor([math.sin(2*math.pi*n*freq*dt) for n in range(nsteps)]).reshape(1,nsteps).repeat([b,1])

  # initial condition
  #if not exciteOn:
    #p[:, excitation_y, excitation_x, 1] = 1
  #excitation_x = w//2
  #excitation_y = h//2
  #p[:, excitation_y, excitation_x, 1] = 1
  p[:,1:h-1,1:w-1,1] = p0 # everywhere but bondary frame

  p_neigh = torch.zeros([b, h, w, 4], device=dev) # last dimension will contain: p now of left, right, top and bottom neighbor, respectively
  bound_neigh = torch.zeros([b, h, w, 4, 2], device=dev) # second last dimension will contain: boundaries info [is wall? and gamma] of left, right, top and bottom neighbor, respectively
  # for now boundaries [and in particular gamma] are static
  bound_neigh[:,0:h,1:w,0,:] = bound[:,0:h,0:w-1,:] # we retrieve boundary info from x-1 [left neighbor]
  bound_neigh[:,0:h,0:w-1,1,:] = bound[:,0:h,1:w,:] # we retrieve boundary info from x+1 [right neighbor]
  bound_neigh[:,1:h,0:w,2,:] = bound[:,0:h-1,0:w,:] # we retrieve boundary info from y-1 [top neighbor] -> y grows from top to bottom!
  bound_neigh[:,0:h-1,0:w,3,:] = bound[:,1:h,0:w,:] # we retrieve boundary info from y+1 [bottom neighbor]

  p_lrtb  = torch.zeros([b, h-1, w-1, 4], device=dev) # temp tensor

  # where to save solutions
  sol = torch.zeros([b,h,w,nsteps], device=dev)
  sol_t = torch.zeros(nsteps, device=dev)

  t=0.0
  for step in range(nsteps):

    # p_next = [ 2*p_now + (mu-1)*p_prev + rho*(pL + pR + pT + pB - 4*p_prev) ] / (mu+1)
    # with pL:
    # p_now_left_neighbor -> if air
    # p_now * 1-gamma_left_neighbor -> if wall [boundary]

    # we can avoid updating the edges, for p on edges is never used!-> [1:h-1,1:w-1,...] instead of [0:h,0:w,...] or [:,:,...]

    # sample neighbors' pressure
    p_neigh[:,1:h-1,1:w-1,0] = p[:,1:h-1,0:w-2,1] # we sample p now on x-1 [left neighbor]
    p_neigh[:,1:h-1,1:w-1,1] = p[:,1:h-1,2:w,1] # we sample p now on x+1 [right neighbor]
    p_neigh[:,1:h-1,1:w-1,2] = p[:,0:h-2,1:w-1,1] # we sample p now on y-1 [top neighbor] -> y grows from top to bottom!
    p_neigh[:,1:h-1,1:w-1,3] = p[:,2:h,1:w-1,1] # we sample p now on y+1 [bottom neighbor]

    # enforce boundary conditions
    # p_lrtb is already sub matrix, no need for indices
    p_lrtb = p_neigh[:,1:h-1,1:w-1,:]*(1-bound_neigh[:,1:h-1,1:w-1,:,0]) # if air
    p_lrtb += bound_neigh[:,1:h-1,1:w-1,:,0]*p[:,1:h-1,1:w-1,1].reshape(b,h-2,w-2,1).repeat([1,1,1,4])*(1-bound_neigh[:,1:h-1,1:w-1,:,1]) # if wall

    p[:,1:h-1,1:w-1,2] = 2*p[:,1:h-1,1:w-1,1] + (prop[:,1:h-1,1:w-1,0]-1)*p[:,1:h-1,1:w-1,0] 
    p[:,1:h-1,1:w-1,2] += prop[:,1:h-1,1:w-1,1]*(torch.sum(p_lrtb,3) - 4*p[:,1:h-1,1:w-1,1]) 
    p[:,1:h-1,1:w-1,2] /= (prop[:,1:h-1,1:w-1,0]+1)

    #if exciteOn:
    #  p[:, excitation_y,excitation_x,2] += excite[:,step]

    # if display, print the time step and plot the frames of the first
    # solution of the batch, at the requested rate
    if disp :
      if (step+1) % disp_delta == 0 :
        print('step', step+1, 'of', nsteps)
        plotDomain(p[0,:,:,2])

    t += dt 

    # save return values
    sol[...,step] = p[...,2]
    sol_t[step] = t

    # update
    p[:,1:h-1,1:w-1,0] = p[:,1:h-1,1:w-1,1]
    p[:,1:h-1,1:w-1,1] = p[:,1:h-1,1:w-1,2]

  return sol, sol_t






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
  print('Config File Not Found at {}'.format(config_path))
  sys.exit()


# read params from config file

# path
dataset_root = Path(config['dataset_generation'].get('path'))

# dataset size
N = config['dataset_generation'].getint('N') # num of dataset entries
B = config['dataset_generation'].getint('B') # batch size

# domain and time parameters
s = config['dataset_generation'].getint('s')
# time parameters
nsteps = config['dataset_generation'].getint('nsteps') # = T_in+T_out, e.g., Tin = 10, T = 10 -> input [0,Tin), output [10, Tin+T)
samplerate = config['dataset_generation'].getint('samplerate'); # Hz, probably no need to ever modify this...

# propagation params, for now common to all grid points
# explained further below
mu = config['dataset_generation'].getfloat('mu') # damping factor, positive and typically way below 1
rho = config['dataset_generation'].getfloat('rho') # 'propagation' factor, positive and lte 0.5; formally defined as rho = [c*ds/dt)]^2, with c=speed of sound in medium, ds=size of each grid point [same on x and y], dt=1/samplerate
gamma = config['dataset_generation'].getfloat('gamma') # type of edge, 0 if clamped edge, 1 if free edge

# checkpoints
cp = config['dataset_generation'].getint('checkpoints') # num of checkpoints

dryrun = config['dataset_generation'].getint('dryrun') # visualize a single simulation run

# excitation parameters
#exciteOn = 0
#excitation_x = w//2
#excitation_y = h//2
#freq = 4400
#-------------------------------------------------------------------------------

# square domain
w = s
h = s

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"

print('device:', dev)




if dryrun == 0 :

  # either Generate Full Dataset and Save it

  DATASET_INDEX = '0'

  time_duration = nsteps/samplerate 
  print('simulation duration:', time_duration, 's')

  # num of checkpoints must be lower than total number of batches
  if cp > N//B :
    cp = (N//B)//2 # a checkpoint every other batch
  if cp == 0 :
    cp = 1

  # output file name and location
  MU = str(mu)
  MU = MU.replace('.', '@')
  RHO = str(rho)
  RHO = RHO.replace('.', '@')
  GAMMA = str(gamma)
  GAMMA = GAMMA.replace('.', '@')

  dataset_name = 'iwe_d'+DATASET_INDEX+'_n'+str(N)+'_t'+str(nsteps)+'_s'+str(s)+'_mu'+MU+'_rho'+RHO+'_gamma'+GAMMA
  dataset_path = dataset_root.joinpath(dataset_name)

  print(dataset_path)


  n_cnt=0
  num_of_batches = N//B
  batches_per_cp = num_of_batches//cp
  cp_size = batches_per_cp * B # num of data points per checkpoint
  cp_cnt = 0
  rem = 0 # is there any remainder?

  # create folder where to save dataset
  dataset_path.mkdir(parents=True, exist_ok=True)


  # compute number of leading zeros for pretty file names
  cp_num = str(cp)
  l_zeros=len(cp_num) # number of leading zeros in file name
  # check if l_zeros needs to be lowered down
  # e.g., cp_num = 100 -> [0, 99] -> should be printed with only one leading zero:
  # 01, 02, ..., 98, 99
  cc = pow(10,l_zeros-1)
  if cp <= cc :
    l_zeros = l_zeros-1 


  t1 = default_timer()

  # initial conditions
  a = torch.zeros(cp_size, h, w)
  # solutions
  u = torch.zeros(cp_size, h, w, nsteps)


  for b in range(num_of_batches):
    # initial conditions
    p0 = torch.randn(B, h-2, w-2) # everywhere but bondary frame
    # compute all steps in full batch
    sol, sol_t = irreducible_wave_equation(dev, B, p0, w, h, 1/samplerate, nsteps, mu, rho, gamma)#, exciteOn, freq, excitation_x, excitation_y)
    # store
    a[n_cnt:(n_cnt+B),1:h-1,1:w-1] = p0 # initial condition
    u[n_cnt:(n_cnt+B),...] = sol # results

    n_cnt += B

    # save some checkpoints, just in case...
    if (b+1) % batches_per_cp == 0 : 
      file_name = dataset_name + '_cp' + str(cp_cnt).zfill(l_zeros) + '_' + str(n_cnt) + '.mat'
      file_name = dataset_path.joinpath(file_name)
      scipy.io.savemat(file_name, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
      print('\tbatch', b+1, 'of', num_of_batches, '(checkpoint '+str(cp_cnt)+', '+str(cp_size)+' dataset points)')
      cp_cnt += 1
      # reset initial conditions, solutions and data point count
      a = torch.zeros(cp_size, h, w)
      u = torch.zeros(cp_size, h, w, nsteps)
      n_cnt = 0
    elif (b+1) == num_of_batches :
      file_name = dataset_name + '_rem_' + str(n_cnt)  + '.mat'
      file_name = dataset_path.joinpath(file_name)
      scipy.io.savemat(file_name, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
      print('\tbatch', b+1, 'of', num_of_batches, '(remainder, '+str(n_cnt)+' dataset points)')
      rem = 1

  t2 = default_timer()
  print('\nDataset', dataset_name, 'saved in:')
  print('\t', dataset_path)
  print('split in', cp_cnt, 'checkpoints with', cp_size, 'datapoints each')
  if rem :
    print('plus remainder file with', n_cnt, 'datapoints')

  simulation_duration = t2-t1
  print('\nElapsed time:', simulation_duration, 's')



  # save relavant bits of general config file + extra info in local dataset config file 
  #TODO define logic for type of dataset/solver and add notes/comments

  # create empty config file
  config = configparser.RawConfigParser()
  config.optionxform = str # otherwise raw config parser converts all entries to lower case letters
  # fill it
  config.add_section('dataset_details')
  config.set('dataset_details', 'N', N)
  config.set('dataset_details', 'B', B)
  config.set('dataset_details', 's', s)
  config.set('dataset_details', 'nsteps', nsteps)
  config.set('dataset_details', 'samplerate', samplerate)
  config.set('dataset_details', 'mu', mu)
  config.set('dataset_details', 'rho', rho)
  config.set('dataset_details', 'gamma', gamma)
  config.set('dataset_details', 'checkpoints', cp)
  config.set('dataset_details', 'simulated_time_s', time_duration)
  config.set('dataset_details', 'simulation_duration_s', simulation_duration)
  # where to write it
  config_path = dataset_path.joinpath(dataset_name+'_config.ini')
  # write
  with open(config_path, 'w') as configfile:
      config.write(configfile)




else :

  # or Generate 1 data entry and visualize it [same content as solver function]"""

  disp_rate = 1/1

  b=1 # 1 entry batch

  # data structures

  # initial conditions
  # everywhere but bondary frame
  p0 = torch.randn(b, h-2, w-2)

  #p0 = torch.zeros(b, h-2, w-2)
  #p0[:,10:30,10:30] = noise[:,:,:]

  #p0 = torch.zeros(b, h-2, w-2)
  #excitation_x = w//2
  #excitation_y = h//2
  #p0[:, excitation_y, excitation_x] = 1

  sol, _ = irreducible_wave_equation(dev, B, p0, w, h, 1/samplerate, nsteps, mu, rho, gamma, True, disp_rate)