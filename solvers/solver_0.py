import torch
from neuralacoustics.data_plotter import plotDomain # to plot dryrun


description = '2D irreducible wave equation solver, for transverse waves (xi = displacement), with static boundaries and acoustic parameters'


# solver
def run(dev, dt, nsteps, b, w, h, mu, rho, gamma, xi0, bnd=torch.empty(0, 1), excite=torch.empty(0, 1), disp=False, dispRate=1):
  # check arguments

  # boundaries passed in
  if bnd.size(dim=0) == 0:
    extraBound = False
  else:
    extraBound = True

  # excitation
  if excite.size(dim=0) == 0:
    exciteOn = False
  else:
    exciteOn = True

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
  if exciteOn is True:
    full_excitation = torch.zeros([b, h, w, nsteps], device=dev) 
    full_excitation[..., excite.size(dim=3)] = excite # embed excitation in bigger tensor that spans full simulation time
  #excite = torch.FloatTensor([math.sin(2*math.pi*n*freq*dt) for n in range(nsteps)]).reshape(1,nsteps).repeat([b,1])

  # initial condition
  xi[:,1:h-1,1:w-1,1] = xi0 # everywhere but bondary frame

  xi_neigh = torch.zeros([b, h, w, 4], device=dev) # last dimension will contain: xi now of left, right, top and bottom neighbor, respectively
  bound_neigh = torch.zeros([b, h, w, 4, 2], device=dev) # second last dimension will contain: boundaries info [is wall? and gamma] of left, right, top and bottom neighbor, respectively
  # for now boundaries [and in particular gamma] are static
  bound_neigh[:,0:h,1:w,0,:] = bound[:,0:h,0:w-1,:] # we retrieve boundary info from x-1 [left neighbor]
  bound_neigh[:,0:h,0:w-1,1,:] = bound[:,0:h,1:w,:] # we retrieve boundary info from x+1 [right neighbor]
  bound_neigh[:,1:h,0:w,2,:] = bound[:,0:h-1,0:w,:] # we retrieve boundary info from y-1 [top neighbor] -> y grows from top to bottom!
  bound_neigh[:,0:h-1,0:w,3,:] = bound[:,1:h,0:w,:] # we retrieve boundary info from y+1 [bottom neighbor]

  xi_lrtb  = torch.zeros([b, h-1, w-1, 4], device=dev) # temp tensor

  # where to save solutions
  sol = torch.zeros([b,h,w,nsteps], device=dev)
  sol_t = torch.zeros(nsteps, device=dev)


  #--------------------------------------------------------------
  # simulation loop 

  t=0.0
  for step in range(nsteps):

    # xi_next = [ 2*xi_now + (mu-1)*xi_prev + rho*(pL + pR + pT + pB - 4*xi_prev) ] / (mu+1)
    # with pL:
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

    if exciteOn:
      xi[:,1:h-1,1:w-1,2] += full_excitation[...,step]


    # if display, print the time step and plot the frames of the first
    # solution of the batch, at the requested rate
    if disp:
      if (step+1) % disp_delta == 0:
        # print first entry in batch
        displacement = xi[0,:,:,2] * (-1*bound[0,:,:,0] + 1) # set zero displacement to boundaries, to identify them
        print(f'step {step+1} of {nsteps}')
        plotDomain(displacement)
 
    t += dt 

    # save return values
    sol[...,step] = xi[...,2]
    sol_t[step] = t

    # update
    xi[:,1:h-1,1:w-1,0] = xi[:,1:h-1,1:w-1,1]
    xi[:,1:h-1,1:w-1,1] = xi[:,1:h-1,1:w-1,2]

  return sol, sol_t




def getDescription():
  return description