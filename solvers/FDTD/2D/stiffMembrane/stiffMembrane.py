import torch
import math # sqrt
from neuralacoustics.data_plotter import plotDomain # to plot dryrun

def run(dev, dt, nsteps, b, w, h, sigma0, sigma1, T, nu, E, H, rho, excite, disp=False, dispRate=1, pause=0):

    # excitation
    full_excitation = torch.zeros([b, h-2, w-2, nsteps+1], device=dev) 
    full_excitation[..., 1:] = excite[...] # copy excitation to tensor on device 
    
    # display
    if disp:
        if dispRate > 1:
            dispRate = 1
        disp_delta = 1/dispRate

    #--------------------------------------------------------------
    # params

    k = dt # time resolution [s]
    #sigma0 = 4 # frequency independent loss coefficient (damping), value taken from p. 49
    #sigma1 = 0.02 # frequency dependent loss coefficient (damping), value taken from p. 49
    #T = 1000 # surface tension [N/m], value taken from p. 137
    #nu = 0.2 # poisson ratio [pure], between 0 and 0.5 (example on p. 17)
    #E = 1000.5*10**9 # Young's modulus [Pa], value taken from p. 162 (other exmaple p. 17)
    #H = 0.0001 # plate's thinkness [m], reasonable value, also on p. 140 and 17
    #rho = 1000 # density of plate [kg/m^3], value taken from p. 163 (other exmaple p. 17)


    #--------------------------------------------------------------
    # derived params

    D = (E*H**3)/( 12*(1-nu**2) ) # plate's bending stiffness, p. 14

    kappa = math.sqrt(D/(rho*H)) # plate coefficient, p. 15, only example value is 1.24 (p. 49)

    c = math.sqrt(T/(rho*H)) # speed of propagation of wave [m/s], should be in the order of 100 (p. 100)

    # spatial resolution [m], obtained via stability condition (p. 49)
    hh = math.sqrt( ((c*k)**2 + 4*sigma1*k)**2 + 16*(kappa*k)**2 )
    hh = math.sqrt( (c*k)**2 + 4*sigma1*k + hh )

    print(f'D: {D}')
    print(f'kappa: {kappa}')
    print(f'c: {c}')
    print(f'hh: {hh}')


    #--------------------------------------------------------------
    # data structures/tensors

    domainW = w
    domainH = h
    # create the frame, contains domain [that includes rim] + extra outside cell layers around, that do not affect the computation
    frameW = domainW+2 # 2 is for layers of outside cells, left and right
    frameH = domainH+2 # 2 is for layers of outside cells, top and bottom

    # VIC temp intial condition
    #ex_x += 2
    #ex_y += 2

    # only inner part of frame will be updated
    updateFrameW = frameW-4 # no outside layers, nor rim!
    updateFrameH = frameH-4 # no outside layers, nor rim!

    # indices to traverse all frame except rim and dead layers
    updateStartH = 2 # skip first dead layer and first rim layer on y
    updateStartW = 2 # skip first dead layer and first rim layer on x
    updateEndH = updateStartH+domainH-2 # skip last rim layer and dead layer on y
    updateEndW = updateStartW+domainW-2 # skip last rim layer and dead layer on x

    
    # updateFrameW is equal to updateEndW-updateStartW
    # and 
    # updateFrameH is equal to updateEndH-updateStartH


    #VIC for now not used
    # type_slice_rim = 0
    # type_slice_out = 1
    # cell_type = torch.zeros([b, frameH, frameW, 2], device=dev) # last dimension contains: is rim? and is outside?
    # # the last layer of cells received as w and h is the rim
    # # we then need to add an extra layer of cells outside the rim [w+1 and h+1], as needed by the stencil
    # # rim cells are same as 'boundary', where boundary conditions are enforced
    # # outside cells are always set to 0 displacement
    # # populate
    # # rim/boundaries
    # cell_type[:,1:-1,1,type_slice_rim] = 1 # right vertical rim
    # cell_type[:,1:-1,frameW-2,type_slice_rim] = 1 # left vertical rim
    # cell_type[:,1,1:-1,type_slice_rim] = 1 # top horizontal rim
    # cell_type[:,frameH-2,1:-1,type_slice_rim] = 1 # bottom horizontal rim
    # # oustide
    # cell_type[:,:,0,type_slice_out] = 1 # rightmost vertical outside layer 
    # cell_type[:,:,frameW-1,type_slice_out] = 1 # leftmost vertical outside layer
    # cell_type[:,0,:,type_slice_out] = 1 # top horizontal outside layer
    # cell_type[:,frameH-1,:,type_slice_out] = 1 # bottom horizontal outside layer
    # # the rest are regular cells on the membrane!

    time_slice_prev = 0
    time_slice_now = 1
    time_slice_next = 2

    # displacement
    w = torch.zeros([b, frameH, frameW, 3], device=dev) # last dimension contains: w prev, w now, w next
    sol = torch.zeros([b, frameH, frameW, nsteps+1], device=dev) # to save solutions

    # current and previous displacements for calulations, only on part of domain that is updated
    wn   = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_1 = torch.zeros([b, updateFrameH, updateFrameW], device=dev)


    # laplacians, on wn and wn-1, only on part of domain that is updated
    nabla_n = torch.zeros([b, updateFrameH, updateFrameW], device=dev) 
    nabla_n_1 = torch.zeros([b, updateFrameH, updateFrameW], device=dev)

    # biharmonic operator on wn, only on part of domain that is updated
    biharm = torch.zeros([b, updateFrameH, updateFrameW], device=dev) 

    # neighbors, only on part of domain that is updated
    wn_l = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_r = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_u = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_d = torch.zeros([b, updateFrameH, updateFrameW], device=dev)

    wn_ll = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_rr = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_uu = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_dd = torch.zeros([b, updateFrameH, updateFrameW], device=dev)

    wn_lu = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_ld = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_ru = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_rd = torch.zeros([b, updateFrameH, updateFrameW], device=dev)

    # fourth order difference gradients, only on part of domain that is updated
    #wn_xxxx = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    #wn_yyyy = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    #wn_xxyy = torch.zeros([b, updateFrameH, updateFrameW], device=dev)

    # previous neighbors, neededs for sigma1 term, only on part of domain that is updated
    wn_1_l = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_1_r = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_1_u = torch.zeros([b, updateFrameH, updateFrameW], device=dev)
    wn_1_d = torch.zeros([b, updateFrameH, updateFrameW], device=dev)

    # backwards time derivative times nabla, only on part of domain that is updated 
    w_t_nabla = torch.zeros([b, updateFrameH, updateFrameW], device=dev)

    #mic_x = 20 audio saving functionality
    #mic_y = 20
    #samples = torch.zeros([nsteps])

    #--------------------------------------------------------------
    # simulation loop 

    # eq. 2.57, p. 19
    # discretized version eq. 2.166, p. 49
    # with explicit update eq. 2.167, p. 49
    for step in range(nsteps):

        # add excitation to w:
        # only adding excitation to the initally specified domain
        w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_now]  += full_excitation[..., step+1] # last dimension contains: w prev, w now, w next
       
        # neighbors, in space and/or time
        wn_l = w[:, updateStartH:updateEndH,      updateStartW-1:updateEndW-1, time_slice_now]
        wn_r = w[:, updateStartH:updateEndH,      updateStartW+1:updateEndW+1, time_slice_now]
        wn_u = w[:, updateStartH-1:updateEndH-1,  updateStartW:updateEndW,     time_slice_now]
        wn_d = w[:, updateStartH+1:updateEndH+1,  updateStartW:updateEndW,     time_slice_now]

        wn_ll = w[:, updateStartH:updateEndH,     updateStartW-2:updateEndW-2, time_slice_now]
        wn_rr = w[:, updateStartH:updateEndH,     updateStartW+2:updateEndW+2, time_slice_now]
        wn_uu = w[:, updateStartH-2:updateEndH-2, updateStartW:updateEndW,     time_slice_now]
        wn_dd = w[:, updateStartH+2:updateEndH+2, updateStartW:updateEndW,     time_slice_now]

        wn_lu = w[:, updateStartH-1:updateEndH-1, updateStartW-1:updateEndW-1, time_slice_now]
        wn_ld = w[:, updateStartH+1:updateEndH+1, updateStartW-1:updateEndW-1, time_slice_now]
        wn_ru = w[:, updateStartH-1:updateEndH-1, updateStartW+1:updateEndW+1, time_slice_now]
        wn_rd = w[:, updateStartH+1:updateEndH+1, updateStartW+1:updateEndW+1, time_slice_now]

        wn =   w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_now]
        wn_1 = w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_prev]

        wn_1_l = w[:, updateStartH:updateEndH,      updateStartW-1:updateEndW-1, time_slice_prev]
        wn_1_r = w[:, updateStartH:updateEndH,      updateStartW+1:updateEndW+1, time_slice_prev]
        wn_1_u = w[:, updateStartH-1:updateEndH-1,  updateStartW:updateEndW,     time_slice_prev]
        wn_1_d = w[:, updateStartH+1:updateEndH+1,  updateStartW:updateEndW,     time_slice_prev]



        # laplacians, on wn and wn-1 (p. 28)
        nabla_n = (wn_l + wn_u - 4*wn + wn_r + wn_d)/hh**2
        nabla_n_1 = (wn_1_l + wn_1_u - 4*wn_1 + wn_1_r + wn_1_d)/hh**2


        # fourth order difference gradients
        #wn_xxxx = (wn_ll - 4*wn_l + 6*wn - 4*wn_r + wn_rr)/hh**4
        #wn_yyyy = (wn_uu - 4*wn_u + 6*wn - 4*wn_d + wn_dd)/hh**4
        #wn_xxyy = (wn_lu - 2*wn_l + wn_ld - 2*wn_u + 4*wn - 2*wn_d + wn_ru - 2*wn_r + wn_rd)/hh**4

        # biharmonic operator on wn (p. 28)
        biharm  = 20*wn
        biharm += wn_ll - 8*wn_l - 8*wn_r + wn_rr
        biharm += wn_uu - 8*wn_u - 8*wn_d + wn_dd
        biharm += 2*wn_lu + 2*wn_ld + 2*wn_ru + 2*wn_rd
        biharm /= hh**4
        # same as
        #biharm = wn_xxxx + 2*wn_xxyy + wn_yyyy 


        # backwards time derivative times nabla 
        w_t_nabla = nabla_n/k - nabla_n_1/k

        # w next
        w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_next] = T*nabla_n - D*biharm + (sigma0/k)*rho*H*wn_1 + 2*sigma1*rho*H*w_t_nabla
        w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_next] *= k**2/(rho*H)
        w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_next] += 2*wn - wn_1
        w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_next] /= 1+sigma0*k
        
        
        if disp:
            if (step+1) % disp_delta == 0:
                # print first entry in batch
                print(f'Step {step+1} of {nsteps} - Max: {w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_next].max().item()}')
                plotDomain(w[0,:,:,time_slice_now], pause=pause)
        
        sol[...,step+1] = w[..., time_slice_next] # save output (future state)


        w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_prev] = w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_now]
        w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_now] = w[:, updateStartH:updateEndH, updateStartW:updateEndW, time_slice_next]
        
        #TODO add BC
        # currently, centered clamp boundary conditions are 'implicitly' enforced
        # by non-updating the rim, i.e., w = 0
        # in accordance with eq 2.134 and 2.137, p. 35
        

        #samples[step] = w[0, mic_y, mic_x, time_slice_next]

    #samples = samples.unsqueeze(0)
    #current_time = datetime.now()
    #current_time = current_time.strftime('%Y-%m-%d_%H:%M:%S')
    #file_name = f"stiffMembrane_{current_time}.wav"
    #torchaudio.save(file_name, samples, math.floor(1/dt))
