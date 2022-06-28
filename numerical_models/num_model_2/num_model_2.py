import torch
import configparser, argparse # to read config from ini file
import math # for pi and floor
from pathlib import Path # to properly handle paths and folders on every os
from neuralacoustics.utils import openConfig


# to store values from load()
solver = 0 # where to load solver
modelName = ''
w = -1
h = -1
mu = -1
rho = -1
gamma = -1

# grid params
num_xFreqs = 0
num_yFreqs = 0
input_grid = [] # all magnitude-phase couples per each x spatial freq will be stored here
# internal state
current_input = 0

# working copy of domain size vars
W = 0
H = 0


def load(model_name, config_path, _w, _h):
    # to prevent python from declaring new local variables with the same names
    # only needed when content of variables is modfied
    global solver 
    global modelName
    global w
    global h
    global mu
    global rho
    global gamma
    global num_xFreqs
    global num_yFreqs
    global input_grid
    global W
    global H

    modelName = model_name

    # get config file
    config = openConfig(config_path, modelName)
    
    w = _w
    h = _h


    #--------------------------------------------------------------

    # read from config file
    # solver
    solver_dir = config['solver'].get('solver_dir')
    solver_name = config['solver'].get('solver_name')

    # simulation parameters
    mu = config['numerical_model_parameters'].getfloat('mu') # damping factor, positive and typically way below 1; defined as ( 1+(eta*dt)/2 )^-1, with eta=viscosity of medium and dt=1/samplerate
    rho = config['numerical_model_parameters'].getfloat('rho') # 'propagation' factor, positive and lte 0.5; defined as rho = [v*ds/dt)]^2, with v=speed of wave in medium, ds=size of each grid point [same on x and y] and dt=1/samplerate
    gamma = config['numerical_model_parameters'].getfloat('gamma') # type of edge, 0 if clamped edge, 1 if free edge

    # grid parameters -> grid refers to how parameters are sampled between a min and a max
    mag_min = config['numerical_model_parameters'].getfloat('mag_min') # minimum magnitude in the grid
    mag_max = config['numerical_model_parameters'].getfloat('mag_max') # maximum magnitude in the grid
    mag_step = config['numerical_model_parameters'].getfloat('mag_step') # step between consectutive magnitudes in grid

    # phase is normalized
    phase_min = config['numerical_model_parameters'].getfloat('phase_min') # minimum phase in the grid
    phase_max = config['numerical_model_parameters'].getfloat('phase_max') # maximum phase in the grid
    phase_step = config['numerical_model_parameters'].getfloat('phase_step') # step between consectutive phases in grid

    bin_min = config['numerical_model_parameters'].getint('bin_min') # index of starting [minimum spatial freq], then grid will span full width of frequency domain


    #--------------------------------------------------------------
    # init input grid, always deterministic

    # prepare input frequencies as mag and phase couples
    mag_num = math.floor((mag_max-mag_min)/mag_step) + 1
    phase_num = math.floor((phase_max-phase_min)/phase_step) + 1

    magnitudes = [mag_min + i*mag_step for i in range(mag_num)]
    phases = [phase_min + i*phase_step for i in range(phase_num)]


    # compute size of frequency domain
    
    W = (w-2) # -2 to remove boundary frame
    H = (h-2) # -2 to remove boundary frame
    # number of bins with positive frequencies in both dimensions
    num_xFreqs = W//2 + 1 
    num_yFreqs = H//2 + 1

    # we need at least 1 bin
    if bin_min >= num_xFreqs:
        bin_min = 0

    num_bins = num_xFreqs - bin_min

    # empty structure
    input_grid = torch.zeros([mag_num*phase_num*num_bins,3]) # first dimension is index of input, then spatial freq/magnitude/phase
    # fill it!
    for x in range(num_bins):
        for m in range(mag_num):
            for p in range(phase_num):
                input_grid[x*mag_num*phase_num + m*phase_num + p, 0] = x+bin_min
                input_grid[x*mag_num*phase_num + m*phase_num + p, 1] = magnitudes[m]
                input_grid[x*mag_num*phase_num + m*phase_num + p, 2] = phases[p]

    #--------------------------------------------------------------------------------------

    # load solver
    # we want to load the package through potential subfolders
    solver_path_folders = Path(solver_dir.replace('PRJ_ROOT', '.')).parts

    # create package structure by concatenating folders with '.'
    packages_struct = solver_path_folders[0]
    for pkg in range(1,len(solver_path_folders)):
        packages_struct += '.'+solver_path_folders[pkg] 
    # load
    solver = __import__(packages_struct + '.' + solver_name, fromlist=['*']) # i.e., all.packages.in.solver.dir.solver_name

    # return number of inputs that the model will span, i.e., number of data points
    return len(input_grid)


def run(dev, dt, nsteps, b, disp=False, dispRate=1, pause=0):
    # to prevent python from declaring new local variables with the same names
    # only needed when content of variables is modfied
    global current_input

    # set parameters

    # propagation params, explained in solver
    # potentially model can have different values across domain
    _mu = torch.ones(b, h, w) * mu
    _rho = torch.ones(b, h, w) * rho
    _gamma = torch.ones(b, h, w) * gamma

    #--------------------------------------------------------------
    # initial condition
    ex_input_freq = torch.zeros(b).long() # spatial freq, as long to be used as index of freq tensor
    ex_input_mag = torch.zeros(b) # magnitude
    ex_input_phase = torch.zeros(b) # phase    

    if current_input+b-1 < len(input_grid):
        ex_input_freq[:] = input_grid[current_input:current_input+b, 0]
        ex_input_mag[:] = input_grid[current_input:current_input+b, 1]
        ex_input_phase[:] = input_grid[current_input:current_input+b, 2]*2*math.pi
    else:
        # fill a partial batch, with as many inputs as we have left
        grid_len = len(input_grid)
        partial_b = grid_len - current_input
        ex_input_freq[:partial_b] = input_grid[current_input:, 0]
        ex_input_mag[:partial_b] = input_grid[current_input:, 1]
        ex_input_phase[:partial_b] = input_grid[current_input:, 2]*2*math.pi
        # once we have exhausted all the possible inputs, we fill the remainder with silence
        rem = b-partial_b
        ex_input_freq[rem:] = 0
        ex_input_mag[rem:] = 0
        ex_input_phase[rem:] = 0

    # advance for next call
    current_input += b


    xi0 = torch.zeros(b, H, W) # this will be overwritten with initial condition
    # create empty spatial frequency tensor, to initialize
    real = torch.zeros(b, H, num_xFreqs)
    imag = torch.zeros(b, H, num_xFreqs)
    freq = torch.complex(real, imag)
    # an alternative solution to switch to frequency domain:
    #freq = torch.fft.rfft2(xi0) # ready to initialize the domain in the spatial frequency domain!

    # spatial frequency initialization
    # polar turns cartesian coords to polar coords, or abs and angle to real and imaginary part
    freq[range(freq.shape[0]), 0, ex_input_freq] = torch.polar(ex_input_mag, ex_input_phase) 
    # tensor indexing with range() instead of ':' adapted from here: https://stackoverflow.com/questions/61096522/pytorch-tensor-advanced-indexing
    # an alternative solution to torch.polar:
    #freq[] = ex_input_mag * exp(1j*ex_input_phase)


    # apply hermitian symmetry to second last dimension, because rfft2 and irfft apply it only to last dimension
    if H % 2 == 0:
        freq[:, num_yFreqs:, :] = torch.flip(freq[:, 1:num_yFreqs-1, :], [1])
    else:
        freq[:, num_yFreqs:, :] = torch.flip(freq[:, 1:num_yFreqs, :], [1]) # no matter what, odd domain sizes will generate weird initial conditions at high freqs
    
    # we go back to space domain
    xi0 = torch.fft.irfft2(freq, xi0[0,...].size()) # pass final H,W size to deal with odd-sized dimensions
    # the resulting waves will never be perfectly symmetric!
    # they represent chuncks of periodic wave and, much like the case of wavetables in synths, their extremes cannot have the same value
    # they actually are a time-step off, generating perfect periodic continuity

    # force normalization of current magnitude value -> needed because we are not taking into account number of bins
    # when turning mag/phase polar coordinates to cartesian Re/Im coordinates
    # element-wise multiplication to tensor slice adapted from here: https://stackoverflow.com/questions/53987906/how-to-multiply-a-tensor-row-wise-by-a-vector-in-pytorch
    xi0 = ex_input_mag[:, None, None] * xi0 / xi0.amax(dim=(1, 2))[:, None, None]
    # notice that despite the initial condition being normalized to the chosen mag value, the maximum displacement during the time simulation
    # will likely go above that due to constructive interference with reflected waves

    excite = torch.zeros(b, h-2, w-2, nsteps) 
    # initial condition is first excitation
    excite[..., 0] = xi0[...]
    #--------------------------------------------------------------


    # run solver
    sol, sol_t = solver.run(dev, dt, nsteps, b, w, h, _mu, _rho, _gamma, excite, torch.empty(0, 1), disp, dispRate, pause)

    return [sol, sol_t]


def getSolverInfo():
    if solver == 0:
        print(f'{model_name}: Cannot get description of solver! Model needs to be run at least once')
        return {'description':  ''}
    return solver.getInfo()
    