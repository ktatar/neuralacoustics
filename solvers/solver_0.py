import torch
import matplotlib.pyplot as plt # to plot dryrun
import matplotlib.colors as mcolors # to potentially use different color colormaps




description = '2D irreducible wave equation solver, with static boundaries and acoustic parameters'




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
def run(dev, dt, nsteps, b, w, h, mu, rho, gamma, p0, walls=torch.empty(0, 1), excite=torch.empty(0, 1), disp=False, dispRate=1) :
  # check arguments

  # walls
  if walls.size(dim=0) == 0 :
    extraWalls = False
  else :
    extraWalls = True

  # excitation
  if excite.size(dim=0) == 0 :
    exciteOn = False
  else :
    exciteOn = True

  # display
  if disp :
    if dispRate > 1 :
      dispRate = 1
    disp_delta = 1/dispRate
  

  #--------------------------------------------------------------
  # data structures

  # propagation 
  prop = torch.zeros([b, h, w, 2], device=dev) # last dimension contains: mu and rho
  prop[:,:,:,0] = mu
  prop[:,:,:,1] = rho

  # boundaries [walls]
  bound = torch.zeros([b, h, w, 2], device=dev) # last dimension contains: is wall [0-no, 1-yes] and gamma [0-clamped edge, 1-free edge] -> wall indices are opposite than Hyper Drumhead case
  # populate 
  if extraWalls :
    bound[:,:,:,0] = walls
  # always create wall edges all around domain [needed for this solver does not include PML]
  bound[:,:,0,0] = 1
  bound[:,:,w-1,0] = 1
  bound[:,0,:,0] = 1
  bound[:, h-1,:,0] = 1
  # assign gamma to walls
  bound[:,:,:,1] = gamma

  # pressure
  p = torch.zeros([b, h, w, 3], device=dev) # last dimension contains: p prev, p now, p next
  # excitation
  if exciteOn :
    full_excitation = torch.zeros([b, h, w, nsteps], device=dev) 
    full_excitation[..., excite.size(dim=3)] = excite # embed excitation in bigger tensor that spans full simulation time
  #excite = torch.FloatTensor([math.sin(2*math.pi*n*freq*dt) for n in range(nsteps)]).reshape(1,nsteps).repeat([b,1])

  # initial condition
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


  #--------------------------------------------------------------
  # simulation loop 

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

    if exciteOn :
      p[:,1:h-1,1:w-1,2] += full_excitation[...,step]


    # if display, print the time step and plot the frames of the first
    # solution of the batch, at the requested rate
    if disp :
      if (step+1) % disp_delta == 0 :
        # print first entry in batch
        pressure = p[0,:,:,2] * (-1*bound[0,:,:,0] + 1) # set zero pressure to walls, to identify them
        print('step', step+1, 'of', nsteps)
        plotDomain(pressure)
 
    t += dt 

    # save return values
    sol[...,step] = p[...,2]
    sol_t[step] = t

    # update
    p[:,1:h-1,1:w-1,0] = p[:,1:h-1,1:w-1,1]
    p[:,1:h-1,1:w-1,1] = p[:,1:h-1,1:w-1,2]

  return sol, sol_t




def getDescription() :
  return description