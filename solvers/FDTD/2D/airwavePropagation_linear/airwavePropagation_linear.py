import torch
import math # sqrt
import numpy as np # array put, arange
from neuralacoustics.data_plotter import plotDomain # to plot dryrun


info = {
  'description': 'temp',
  'mu': 'damping factor, positive and typically way below 1; defined as mu = (eta*dt)/2, with eta=dynamic viscosity of medium and dt=1/samplerate',
  'rho':  '\"propagation\" factor, positive and lte 0.5; defined as rho = [v*(ds/dt)]^2, with v=speed of wave in medium [also sqrt(tension/area density)], ds=size of each grid point/cell [same on x and y] and dt=1/samplerate',
  'gamma': 'type of boundary: 0 if clamped edge, 1 if free edge',
  'n_solutions': 3
}

# solver
def run(dev, dt, nsteps, b, w, h, c, rho, mu, srcDir, exciteV, walls, pmls= 6, pmlAttn = 0.5, disp = False, dispRate = 1, pause = 0):
    
 
    #---------------------------------------------------------------------
    #VIC here starts the actual solver

    # display
    if disp:
        if dispRate > 1:
            dispRate = 1
        disp_delta = 1/dispRate


    # either 2 layers, or nothing 
    if pmls<2:
        pmls = 0

    srate = 1/dt               # Sample rate
    maxSigmaDt = pmlAttn       # Attenuation coefficient at the PML layer
    # useful constant for wall absorption
    z_inv = mu/(rho*c)
    # Spatial resolution, same along x- and y-direction: CFL Condition
    ds = dt*c*math.sqrt(2.0)   
    # Bulk modulus
    kappa = rho*c*c 

    # this stuff MUST STAY
    # values from 0 to 3 shuld be returned as a map via a function
    # define cell type
    cell_wall       = 0
    cell_air        = 1
    cell_excitation = 2
    cell_dead       = 3
    # automatically create all the cell types needed for the requested number of pml layers
    cell_pml = np.empty(pmls)
    fill = np.arange(cell_dead+1, cell_dead+pmls+1)
    ind = np.arange(len(cell_pml))
    np.put(cell_pml, ind, fill)

    cell_numTypes   = cell_dead+pmls+1
    #---------------------------------------------------------------------


    # tensors

    #VIC convenient while translating code
    domainW = w
    domainH = h
    # Create the frame, contains domain + PML + extra dead cell layers around, that do not affect the computation
    frameW = domainW+2*pmls+2 # 2 is for layers of dead cells, left and right
    frameH = domainH+2*pmls+2 # 2 is for layers of dead cells, top and bottom

    # only inner part of frame will be updated
    updateFrameW = frameW-2 # no dead layers
    updateFrameH = frameH-2 # no dead layers


    # indices to traverse all frame except dead layers
    updateStart = 1 # skip first dead layer on x or y
    updateEndH = updateStart+domainH+2*pmls # skip last dead layer on y
    updateEndW = updateStart+domainW+2*pmls # skip last dead layer on x

    # indices to traverse all domain within frame
    domainStart = 1+pmls # skip first dead layer on x or y
    domainEndH = domainStart+domainH # skip last dead layer on y
    domainEndW = domainStart+domainW # skip last dead layer on x




    pos_slice_cntr  = 0 # current cell
    pos_slice_right = 1 # right neighbor
    pos_slice_top   = 2 # top neighbor
    pos_slice_num  = 3

    srcDir_slice_l   = 0
    srcDir_slice_d   = 1
    srcDir_slice_r   = 2
    srcDir_slice_u   = 3
    srcDir_slice_num = 4

    # actual src tensor
    sourceDirection = torch.zeros([b, updateFrameH, updateFrameW, pos_slice_num, srcDir_slice_num], device=dev)
    # sourceDirection index mean: 
    #                          1 = Left  = 1
    #                          2 = Down  = 1
    #                          3 = Right = 1
    #                          4 = Up    = 1
    # embed argument in tensor
    sourceDirection[:, pmls:pmls+domainH, pmls:pmls+domainW, pos_slice_cntr, :] = srcDir
    sourceDirection[:, pmls:pmls+domainH, pmls:pmls+domainW-1, pos_slice_right, :] = sourceDirection[:, pmls:pmls+domainH, pmls+1:pmls+domainW, pos_slice_cntr, :] # last column in domain has no right neighbor
    sourceDirection[:, pmls+1:pmls+1+domainH-1, pmls:pmls+domainW, pos_slice_top, :] = sourceDirection[:, pmls:pmls+domainH-1, pmls:pmls+domainW, pos_slice_cntr, :] # first row in domain has no top neighbor



    # helper vars to access PV_N tensor
    pv_n_pos_slice_p    = 0 # local pressure
    pv_n_pos_slice_vx   = 1 # local horizontal velocity
    pv_n_pos_slice_vy   = 2 # local vertical velocity
    pv_n_pos_slice_t    = 3 # type of cell
    pv_n_pos_slice_num  = 4
    # pressure, velocities and cell type tensor
    PV_N = torch.zeros([b, frameH, frameW, pv_n_pos_slice_num], device=dev)

    # embed types in tensor
    # strat from all air
    PV_N[:, domainStart:domainEndH, domainStart:domainEndW, pv_n_pos_slice_t] = cell_air

    walls = walls+1
    walls[walls==1] = cell_air
    walls[walls==2] = cell_wall

    # extract sources' positions and combine with walls
    srcDir[..., 0] = (srcDir[..., 0]+srcDir[..., 1]+srcDir[..., 2]+srcDir[..., 3])
    walls[srcDir[..., 0]>0] = cell_excitation


    PV_N[:, domainStart:domainEndH, domainStart:domainEndW, pv_n_pos_slice_t] = walls

    # excitaiton tensor
    excitationV = torch.zeros([b, updateFrameH, updateFrameW, nsteps+1], device=dev)
    excitationV[:, pmls:pmls+domainH, pmls:pmls+domainW, 1:] = exciteV


    # To store beta(tube wall) and sigmaPrimedt(PML Layers) for each type of cell 
    # Beta for air & pmls = 1 and for tubewall = 0
    # sigmaPrimedt = sigmaPrime*dt
    # sigmaPrime = 1 - Beta + Sigma
    # e.g - 
    # Sigma=0 for all the non-PML layers. Hence, sigmaPrime = 1 - Beta inside
    # the domain.
    # WALL -> beta = 0, sigma_prima*dt = (1-0)*dt = 1*dt = dt 
    # AIR  -> beta = 1, sigma_prima*dt = (1-1)*dt = 0*dt = 0
    # [NOTE] - We are considering excitation cell as a special wall cell
    sigmadt = torch.zeros([pmls, 1], device=dev)
    typeValues = torch.zeros([cell_numTypes,2], device=dev)
    typeValues[cell_wall,:]       = torch.tensor([0, dt], device=dev)       # wall
    typeValues[cell_air,:]        = torch.tensor([1, 0], device=dev)        # air
    typeValues[cell_excitation,:] = torch.tensor([0, dt], device=dev)       # excitation, like wall
    typeValues[cell_dead, :]      = torch.tensor([0, 1], device=dev)
    #typeValues[cell_dead, :]      = torch.tensor([1, 1000000], device=dev)  # dead cell, no pressure, nor velocity #TODO review this logic, seems wrong
    # huge sigma seems appropriate only if beta = 1 [air] -> because sigma denominator kills velocity to and from neighbor [this impacts a bit the neighbor, whose p is very low though, because it's external pml]
    # if beta = 0 [wall], then sigma cancels out and thne dead cell behaves like wall, with velocity to and from neighbor depending on neighbor's p [which is very low though, because it's external pml]

    # Define sigma for pmls 
    for pmlCounter in range(pmls):
        sigmadt[pmlCounter] = (pmlCounter/(pmls-1)) * maxSigmaDt
        typeValues[int(cell_pml[0]+pmlCounter), :] = torch.tensor([1, sigmadt[pmlCounter]], device=dev)


    if pmls >= 2:
        # Define Dead Cells and PML Layer Cells

        # put single layer of dead cells [no pressure, no velocity] all around frame
        PV_N[:, 0:frameH, 0, pv_n_pos_slice_t]        = cell_dead
        PV_N[:, 0:frameH, frameW-1, pv_n_pos_slice_t] = cell_dead
        PV_N[:, 0, 0:frameW, pv_n_pos_slice_t]        = cell_dead
        PV_N[:, frameH-1, 0:frameW, pv_n_pos_slice_t] = cell_dead

        # Define horizontal PML layers - Start from the outer layers
        # -----Activate horizontal PML Layers-------
        cellType_idx = pmls-1 # start from outermost pml layer
        cellType = cell_pml[cellType_idx]
        fillStart = 1 # skip dead layer
        fillEnd = -1 # skip dead layer
        layer_idx = 1 # skip dead layer
        for pmlCount in range(pmls):   
            PV_N[:, fillStart:fillEnd, layer_idx, pv_n_pos_slice_t]    = cellType    # left vertical layers
            PV_N[:, fillStart:fillEnd, -layer_idx-1, pv_n_pos_slice_t] = cellType    # right vertical layers
            PV_N[:, layer_idx, fillStart:fillEnd, pv_n_pos_slice_t]    = cellType    # top horizontal layers
            PV_N[:, -layer_idx-1, fillStart:fillEnd, pv_n_pos_slice_t] = cellType    # bottom horizontal layers
            # shrink layer
            fillStart = fillStart+1 
            fillEnd = fillEnd-1
            # move layer inwards
            layer_idx = layer_idx+1
            # change layer type
            cellType_idx = cellType_idx-1 # decrease PML
            cellType = cell_pml[cellType_idx]
    else:
        # if no pmls, then put wall all around
        PV_N[:, 0:frameH, 0, pv_n_pos_slice_t]        = cell_wall
        PV_N[:, 0:frameH, frameW-1, pv_n_pos_slice_t] = cell_wall
        PV_N[:, 0, 0:frameW, pv_n_pos_slice_t]        = cell_wall
        PV_N[:, frameH-1, 0:frameW, pv_n_pos_slice_t] = cell_wall



    # now that all cells have a type, we can assign them appropriate betas and sigmaPrimeDts
    # we use a new tensor for that
    types = torch.zeros([b, updateFrameH, updateFrameW, pos_slice_num], device=dev)
    types[..., pos_slice_cntr]  = PV_N[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_t]     # type of current
    types[..., pos_slice_right] = PV_N[:, updateStart:updateEndH, updateStart+1:updateEndW+1, pv_n_pos_slice_t] # type of right neighbor
    types[..., pos_slice_top]   = PV_N[:, updateStart-1:updateEndH-1, updateStart:updateEndW, pv_n_pos_slice_t] # type of top neighbor

    beta = torch.zeros([b, updateFrameH, updateFrameW, pos_slice_num], device=dev) # frameX-2 = skip 2 dead layers; last dim is current, right and top neigh
    beta[..., pos_slice_cntr]  = typeValues[types[..., pos_slice_cntr].long(), 0]  # values for current
    beta[..., pos_slice_right] = typeValues[types[..., pos_slice_right].long(), 0] # values for right neighbor
    beta[..., pos_slice_top]   = typeValues[types[..., pos_slice_top].long(), 0]   # values for top neighbor

    sigmaPrimeDt = torch.zeros([b, updateFrameH, updateFrameW, pos_slice_num], device=dev) # frameX-2 = skip 2 dead layers; last dim is current, right and top neigh
    sigmaPrimeDt[..., pos_slice_cntr]  = typeValues[types[..., pos_slice_cntr].long(), 1]  # values for current
    sigmaPrimeDt[..., pos_slice_right] = typeValues[types[..., pos_slice_right].long(), 1] # values for right neighbor
    sigmaPrimeDt[..., pos_slice_top]   = typeValues[types[..., pos_slice_top].long(), 1]   # values for top neighbor


    dir_slice_x = 0
    dir_slice_y = 1
    dir_slice_num = 2

    # constant caching tensors to speed up computation
    # span all frame except dead layers
    minBeta = torch.zeros([b, updateFrameH, updateFrameW, dir_slice_num], device=dev) # last dim is min beta between current cell and 2 different neighbors
    maxSigmaPrimeDt  = torch.zeros([b, updateFrameH, updateFrameW, dir_slice_num], device=dev) # last dim is max sigmaPrimeDt between current cell and 2 different neighbors
    # in particular, last dimensions are min/max between current cell and right neihbor and min/max between current cell and top neighbor
    minBeta[..., dir_slice_x] = torch.minimum(beta[..., pos_slice_cntr], beta[..., pos_slice_right]) # check against right neighbor
    minBeta[..., dir_slice_y] = torch.minimum(beta[..., pos_slice_cntr], beta[..., pos_slice_top])   # check against top neighbor
    maxSigmaPrimeDt[..., dir_slice_x] = torch.maximum(sigmaPrimeDt[..., pos_slice_cntr], sigmaPrimeDt[..., pos_slice_right]) # check against right neighbor
    maxSigmaPrimeDt[..., dir_slice_y] = torch.maximum(sigmaPrimeDt[..., pos_slice_cntr], sigmaPrimeDt[..., pos_slice_top])   # check against top neighbor

    betasqrDt_invRhoDs = torch.zeros([b, updateFrameH, updateFrameW, 2], device=dev) # last dim is vertical and horizontal direction
    betasqrDt_invRhoDs[..., dir_slice_x] = (minBeta[..., 0]*minBeta[..., 0]*dt)/(rho*ds)
    betasqrDt_invRhoDs[..., dir_slice_y] = (minBeta[..., 1]*minBeta[..., 1]*dt)/(rho*ds)

    rhoCsqrDt_invDs = (kappa*dt)/ds #TODO make tensor if c [hence kappa] is made tensor

    # cache excitation helper tensors

    # Check whether the current cell is an excitation cell, or if right and top neighbors are [in case they spit backwards!]
    # spans all frame except dead layers
    is_excitation = torch.zeros([b, updateFrameH, updateFrameW, pos_slice_num], device=dev) # last dim is curent cell, right and top neighbor
    is_excitation[..., pos_slice_cntr]  = (types[..., pos_slice_cntr] == cell_excitation) # current is excitaiton
    is_excitation[..., pos_slice_right] = (types[..., pos_slice_right] == cell_excitation) # right neighbor is excitaiton
    is_excitation[..., pos_slice_top]   = (types[..., pos_slice_top] == cell_excitation) # top neighbor is excitation

    # combined check that we are not excitations
    # spans all frame except dead layers
    are_we_not_excitations = torch.zeros([b, updateFrameH, updateFrameW, dir_slice_num], device=dev) # last dim is curent cell & right, current cell & top neighbor
    are_we_not_excitations[..., dir_slice_x] = 1-is_excitation[..., pos_slice_cntr] * 1-is_excitation[..., pos_slice_right]
    are_we_not_excitations[..., dir_slice_y] = 1-is_excitation[..., pos_slice_cntr] * 1-is_excitation[..., pos_slice_top]

    # Compute the excitation weight
    excitation_weight = torch.zeros([b, updateFrameH, updateFrameW, 2], device=dev)
    excitation_weight[..., dir_slice_x] = is_excitation[..., pos_slice_cntr]*sourceDirection[..., pos_slice_cntr, srcDir_slice_r] + is_excitation[..., pos_slice_right]*-sourceDirection[..., pos_slice_right, srcDir_slice_l] # either current is horizontal excitation or right neighbor is backwards horizontal excitation
    excitation_weight[..., dir_slice_y] = is_excitation[..., pos_slice_cntr]*sourceDirection[..., pos_slice_cntr, srcDir_slice_u] + is_excitation[..., pos_slice_top]*-sourceDirection[..., pos_slice_top, srcDir_slice_d]   # either current is vertical excitation or top neighbor is backwards vertical excitation

    # if my neighbor is a backwards excitation [i.e., excitation_weight==-1], then copy their excitation here
    excitationV[:, :, :-1, :] = excitationV[:, :, :-1, :]+excitationV[:, :, 1:, :]*(excitation_weight[:, :, :-1, dir_slice_x]==-1).unsqueeze(3).repeat(1, 1, 1, nsteps+1)
    excitationV[:, 1:, :, :] = excitationV[:, 1:, :, :]+excitationV[:, :-1, :, :]*(excitation_weight[:, 1:, :, dir_slice_y]==-1).unsqueeze(3).repeat(1, 1, 1, nsteps+1)

    # caches whether neighbors are air or not
    is_normal_dir = torch.zeros([b, updateFrameH, updateFrameW, 4], device=dev)
    is_normal_dir[..., 0] = (beta[..., pos_slice_right] != typeValues[cell_air, 0]) # is right neighbor not air?
    is_normal_dir[..., 1] = (beta[..., pos_slice_top] != typeValues[cell_air, 0]) # is top neighbor not air?
    is_normal_dir[..., 2] = (beta[..., pos_slice_right] == typeValues[cell_air, 0]) # is right neighbor air? opposite of slice 0
    is_normal_dir[..., 3] = (beta[..., pos_slice_top] == typeValues[cell_air, 0]) # is top neighbor air? opposite of slice 1

    # caches xor between betas of neighbors, needed for wall logic
    xor_term = torch.zeros([b, updateFrameH, updateFrameW, 4], device=dev)
    xor_term[..., 0] = beta[..., pos_slice_right] * (1-beta[..., pos_slice_cntr])
    xor_term[..., 1] = beta[..., pos_slice_cntr] * (1-beta[..., pos_slice_right])
    xor_term[..., 2] = beta[..., pos_slice_top] * (1-beta[..., pos_slice_cntr])
    xor_term[..., 3] = beta[..., pos_slice_cntr] * (1-beta[..., pos_slice_top])

    # caches normals towards walls
    N = torch.zeros([b, updateFrameH, updateFrameW, 4], device=dev)
    N[..., 0] =  0.707106*is_normal_dir[..., 3] + (1-is_normal_dir[..., 3])
    N[..., 1] =  0.707106*is_normal_dir[..., 1] + (1-is_normal_dir[..., 1])
    N[..., 2] =  0.707106*is_normal_dir[..., 2] + (1-is_normal_dir[..., 2])
    N[..., 3] =  0.707106*is_normal_dir[..., 0] + (1-is_normal_dir[..., 0])



    #--------------------------------------------------------------------------------
    # useful tmp tensors
    PV_Nplus1 = torch.zeros([b, frameH, frameW, 3], device=dev) # only P, Vx and Vy [no types, all cached already]
    # no dead cells [because they will not be processed]
    #TODO make single tensor and parallelize
    CxVx      = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    CyVy      = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    #TODO make single tensor and parallelize
    CxP       = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    CyP       = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    #TODO make single tensor and parallelize
    vb_ex    = torch.zeros([b, updateFrameH, updateFrameW, dir_slice_num], device=dev) # last dim is x and y directions
    vb_alpha = torch.zeros([b, updateFrameH, updateFrameW, dir_slice_num], device=dev) # last dim is x and y directions

    # where to save solutions
    #TODO change this logic:
    # _turn solution into output 
    # _and add input -> prev vel+curren excitation [velocity on excitations cells is zero if no input (:]
    # _no need for nsteps+1 anymore
    # this means that:
    # _mat files should contain extra entry [input -> a]
    # _dataste loader logic must be changed, so that T_in steps are taken from a[] and T_out from u[]
    # __nesteps+1 should be removed from dataset loader and dateset_generate [log] too 
    sol = torch.zeros([b, updateFrameH, updateFrameW, 3, nsteps+1], device=dev)
      

    t=0.0
    for step in range(nsteps):
        # STEP1: Calculate (del.V) = (dVx/ds + dVy/ds)
        # CxVx = dVx/ds, where Vx = velocity along x direction
        # CyVy = dVy/ds, where Vy = velocity along y direction
        # spatial differential is bulked in cached const
        CxVx = PV_N[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vx] - PV_N[:, updateStart:updateEndH, updateStart-1:updateEndW-1, pv_n_pos_slice_vx] # Vx - Vx_left                        
        CyVy = PV_N[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vy] - PV_N[:, updateStart+1:updateEndH+1, updateStart:updateEndW, pv_n_pos_slice_vy] # Vy - Vy_down

        # STEP2: Calculate Pr_next                           
        PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_p] = ( PV_N[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_p] - rhoCsqrDt_invDs*(CxVx+CyVy) ) / (1 + sigmaPrimeDt[..., pos_slice_cntr])

        # STEP3: Calculate Vx & Vy
        # To compute Vx we need calculate CxP = (del.P) = dPx/ds
        # To compute Vy we need calculate CyP = (del.P) = dPy/ds
        # spatial differential is bulked in cached const
        CxP = ( PV_Nplus1[:, updateStart:updateEndH, updateStart+1:updateEndW+1, pv_n_pos_slice_p] - PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_p] ) # P_right - P_current
        CyP = ( PV_Nplus1[:, updateStart-1:updateEndH-1, updateStart:updateEndW, pv_n_pos_slice_p] - PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_p] ) # P_top - P_current

        PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vx] = ( minBeta[..., dir_slice_x]*PV_N[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vx] - (betasqrDt_invRhoDs[..., dir_slice_x]*CxP) )
        PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vy] = ( minBeta[..., dir_slice_y]*PV_N[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vy] - (betasqrDt_invRhoDs[..., dir_slice_y]*CyP) )

        # STEP4: excitation
        # Inject the source to the Vx_next and Vy_next = excitationV(T)
        vb_ex[..., dir_slice_x] = excitationV[..., step+1]*excitation_weight[..., dir_slice_x]
        vb_ex[..., dir_slice_y] = excitationV[..., step+1]*excitation_weight[..., dir_slice_y]
        
        # STEP 5: wall absorption
        # Compute vb_alpha
        vb_alpha[..., dir_slice_x] = xor_term[..., 1]*PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_p]*N[..., 1] - xor_term[..., 0]*PV_Nplus1[:, updateStart:updateEndH, updateStart+1:updateEndW+1, pv_n_pos_slice_p]*N[..., 0]
        vb_alpha[..., dir_slice_y] = xor_term[..., 3]*PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_p]*N[..., 3] - xor_term[..., 2]*PV_Nplus1[:, updateStart-1:updateEndH-1, updateStart:updateEndW, pv_n_pos_slice_p]*N[..., 2]

        vb_alpha = vb_alpha*are_we_not_excitations*z_inv

        # STEP 6
        # Update Vx and Vy
        PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vx] = PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vx] + maxSigmaPrimeDt[..., dir_slice_x] * (vb_ex[..., 0] + vb_alpha[..., dir_slice_x])
        PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vy] = PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vy] + maxSigmaPrimeDt[..., dir_slice_y] * (vb_ex[..., 1] + vb_alpha[..., dir_slice_y])

        PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vx] = PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vx]/(minBeta[..., dir_slice_x]+maxSigmaPrimeDt[..., dir_slice_x])
        PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vy] = PV_Nplus1[:, updateStart:updateEndH, updateStart:updateEndW, pv_n_pos_slice_vy]/(minBeta[..., dir_slice_y]+maxSigmaPrimeDt[..., dir_slice_y])

        # STEP 7: Copy PV_Nplus1 to PV_N for the next time step
        PV_N[..., :-1] = PV_Nplus1

        # if display, print the time step and plot the frames of the first
        # solution of the batch, at the requested rate
        if disp:
          if (step+1) % disp_delta == 0:
            # print first entry in batch
            pressure = PV_N[..., pv_n_pos_slice_p]
            pressure[PV_N[..., pv_n_pos_slice_t]==cell_wall] = 100 # set high constant pressure on walls, to identify them
            print(f'step {step+1} of {nsteps}')
            plotDomain(pressure[0, updateStart:updateEndH, updateStart:updateEndW], pause=pause)

        # sol tensor contains 3 solutions [..., a] where a = 0 is pressure, 1 is x velocity,  2 is y velocity
        sol[...,step+1] = PV_N[:, updateStart:updateEndH, updateStart:updateEndW, 0:3]
               
        t += dt
        
    return excitationV[:, pmls:-pmls, pmls:-pmls,:], sol[:, pmls:-pmls, pmls:-pmls,:]


def getInfo():
  return info
