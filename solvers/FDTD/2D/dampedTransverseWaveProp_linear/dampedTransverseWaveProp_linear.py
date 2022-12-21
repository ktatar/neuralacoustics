import torch
from neuralacoustics.data_plotter import plotDomain # to plot dryrun


info = {
  'description': '2D explicit solver of irreducible wave equation, for transverse waves (xi = displacement), with static boundaries and acoustic parameters',
  'mu': 'damping factor, positive and typically way below 1; defined as mu = (eta*dt)/2, with eta=dynamic viscosity of medium and dt=1/samplerate',
  'rho':  '\"propagation\" factor, positive and lte 0.5; defined as rho = [v*(dt/ds)]^2, with v=speed of wave in medium [also sqrt(tension/area density)], ds=size of each grid point/cell [same on x and y] and dt=1/samplerate',
  'gamma': 'type of boundary: 0 if clamped edge, 1 if free edge'
}

# solver
def run(dev, dt, nsteps, b, w, h, mu, rho, gamma, excite, bnd=torch.empty(0, 1), disp=False, dispRate=1, pause=0):
  # check arguments

  # boundaries passed in
  if bnd.size(dim=0) == 0:
    extraBound = False
  else:
    extraBound = True


  # display
  if disp:
    if dispRate > 1:
      dispRate = 1
    disp_delta = 1/dispRate
  

  #--------------------------------------------------------------
  # data structures

  # propagation 
  prop = torch.zeros([b, h, w, 2], device=dev) # last dimension contains: mu and rho
  prop[:,:,:,0] = mu 
  prop[:,:,:,1] = rho 

  # boundaries
  bound = torch.zeros([b, h, w, 2], device=dev) # last dimension contains: is wall [0-no, 1-yes] and gamma [0-clamped edge, 1-free edge] -> wall indices are opposite than Hyper Drumhead case
  # populate 
  if extraBound is True:
    bound[:,:,:,0] = bnd
  # always create wall edges all around domain [needed for this solver does not include PML]
  bound[:,:,0,0] = 1
  bound[:,:,w-1,0] = 1
  bound[:,0,:,0] = 1
  bound[:, h-1,:,0] = 1
  # assign gamma to boundaries
  bound[:,:,:,1] = gamma

  # displacement
  xi = torch.zeros([b, h, w, 3], device=dev) # last dimension contains: xi prev, xi now, xi next

  # excitation
  full_excitation = torch.zeros([b, h-2, w-2, nsteps+1], device=dev) 
  full_excitation[..., 1:] = excite[...] # copy excitation to tensor on device 

  xi_neigh = torch.zeros([b, h, w, 4], device=dev) # last dimension will contain: xi now of left, right, top and bottom neighbor, respectively
  bound_neigh = torch.zeros([b, h, w, 4, 2], device=dev) # second last dimension will contain: boundaries info [is wall? and gamma] of left, right, top and bottom neighbor, respectively
  # for now boundaries [and in particular gamma] are static
  bound_neigh[:,0:h,1:w,0,:] = bound[:,0:h,0:w-1,:] # we retrieve boundary info from x-1 [left neighbor]
  bound_neigh[:,0:h,0:w-1,1,:] = bound[:,0:h,1:w,:] # we retrieve boundary info from x+1 [right neighbor]
  bound_neigh[:,1:h,0:w,2,:] = bound[:,0:h-1,0:w,:] # we retrieve boundary info from y-1 [top neighbor] -> y grows from top to bottom!
  bound_neigh[:,0:h-1,0:w,3,:] = bound[:,1:h,0:w,:] # we retrieve boundary info from y+1 [bottom neighbor]

  xi_lrtb  = torch.zeros([b, h-1, w-1, 4], device=dev) # temp tensor

  # where to save solutions
  sol = torch.zeros([b, h, w, nsteps+1], device=dev)
  # nsteps+1 is the total duration of simulation -> initial condition+requested steps

  #sol[:, 1:h-1, 1:w-1, 0] = full_excitation[..., 0] # xi0, initial condition

  # if we only have an initial condition, apply it once (xi0)
  #if not exciteOn:
  #  xi[:,1:h-1,1:w-1,1] = full_excitation[..., 0] # xi0 everywhere but bondary frame

  #VIC note that, regardless of whether we are using an initial condition or a continuous excitation, 
  # at the first simulation step the 'previous' xi is always all zero! this smooths out a bit the effect of an initial condition


  #--------------------------------------------------------------
  # simulation loop 

  for step in range(nsteps):


    # if we have a continuous excitation over time, keep appying it
    # at the first simulation step the 'previous' xi is always all zero! this smooths out a bit the effect of an initial condition
    xi[:,1:h-1,1:w-1,1] += full_excitation[..., step+1] # at first step, this is xi0, initial condition

    # xi_next = [ 2*xi_now + (mu-1)*xi_prev + rho*(xiL + xiR + xiT + xiB - 4*xi_prev) ] / (mu+1)
    # with xiL:
    # xi_now_left_neighbor -> if air
    # xi_now * 1-gamma_left_neighbor -> if boundary

    # we can avoid updating the edges, for xi on edges is never used!-> [1:h-1,1:w-1,...] instead of [0:h,0:w,...] or [:,:,...]

    # sample neighbors' displacement
    xi_neigh[:,1:h-1,1:w-1,0] = xi[:,1:h-1,0:w-2,1] # we sample xi now on x-1 [left neighbor]
    xi_neigh[:,1:h-1,1:w-1,1] = xi[:,1:h-1,2:w,1] # we sample xi now on x+1 [right neighbor]
    xi_neigh[:,1:h-1,1:w-1,2] = xi[:,0:h-2,1:w-1,1] # we sample xi now on y-1 [top neighbor] -> y grows from top to bottom!
    xi_neigh[:,1:h-1,1:w-1,3] = xi[:,2:h,1:w-1,1] # we sample xi now on y+1 [bottom neighbor]

    # enforce boundary conditions
    # xi_lrtb is already sub matrix, no need for indices
    xi_lrtb = xi_neigh[:,1:h-1,1:w-1,:]*(1-bound_neigh[:,1:h-1,1:w-1,:,0]) # if air
    xi_lrtb += bound_neigh[:,1:h-1,1:w-1,:,0]*xi[:,1:h-1,1:w-1,1].reshape(b,h-2,w-2,1).repeat([1,1,1,4])*(1-bound_neigh[:,1:h-1,1:w-1,:,1]) # if wall

    xi[:,1:h-1,1:w-1,2] = 2*xi[:,1:h-1,1:w-1,1] + (prop[:,1:h-1,1:w-1,0]-1)*xi[:,1:h-1,1:w-1,0] 
    xi[:,1:h-1,1:w-1,2] += prop[:,1:h-1,1:w-1,1]*(torch.sum(xi_lrtb,3) - 4*xi[:,1:h-1,1:w-1,1]) 
    xi[:,1:h-1,1:w-1,2] /= (prop[:,1:h-1,1:w-1,0]+1)

    # if display, print the time step and plot the frames of the first
    # solution of the batch, at the requested rate
    if disp:
      if (step+1) % disp_delta == 0:
        # print first entry in batch
        displacement = xi[0,:,:,1] * (-1*bound[0,:,:,0] + 1) # set zero displacement to boundaries, to identify them
        print(f'step {step+1} of {nsteps}')
        plotDomain(displacement, pause=pause)


    # save return values
    sol[...,step+1] = xi[...,2] # future state (output)
    
    # update
    xi[:,1:h-1,1:w-1,0] = xi[:,1:h-1,1:w-1,1]
    xi[:,1:h-1,1:w-1,1] = xi[:,1:h-1,1:w-1,2]

  return  full_excitation, sol


def getInfo():
  return info
