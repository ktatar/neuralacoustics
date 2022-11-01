import torch
import math # for pi and floor
from pathlib import Path # to properly handle paths and folders on every os
from neuralacoustics.utils import openConfig
from neuralacoustics.utils import import_file

# to store values from load()
solver = 0 # where to load solver
modelName = ''
w = -1
h = -1
mu = torch.empty((1,)) # declared as tensor to facilitate vector operations
rho = torch.empty((1,))
gamma = torch.empty((1,))
ex_input_freq = torch.empty((1,), dtype = torch.long) # declared as long in order to allow indexing
ex_input_mag = torch.empty((1,)) 
ex_input_phase = torch.empty((1,))
dt = -1
nsteps = -1


def load(config_path, prj_root):    
    global modelName

    # get config file
    modelName = Path(__file__).stem #extracts modeName from this filename, stores it in global variable
    config = openConfig(config_path, modelName) 

    #--------------------------------------------------------------

    # read from config file
    solver = config['solver'].get('solver')#solver path
    
    # other params will be passed to run() at run-time

    #--------------------------------------------------------------

    # load
    _load(solver, prj_root, config_path)  # loads solver

    return

def load_test(config_path, prj_root):    
    # to prevent python from declaring new local variables with the same names
    # only needed when content of variables is modified
    global modelName
    global w
    global h
    global mu
    global rho
    global gamma
    global dt
    global nsteps
    global ex_input_freq
    global ex_input_mag
    global ex_input_phase

    
    # get config file
    modelName = Path(__file__).stem #extracts modeName from this filename, stores it in global variable
    config = openConfig(config_path, modelName) 

    #--------------------------------------------------------------

    # read from config file
    # solver
    solver_path = config['solver'].get('solver')

    #parses all simulation parameters
    mu[0] = config['numerical_model_parameters'].getfloat('mu') # damping factor, positive and typically way below 1
    rho[0] = config['numerical_model_parameters'].getfloat('rho') # 'propagation' factor, positive and lte 0.5; formally defined as rho = [c*ds/dt)]^2, with c=speed of sound in medium, ds=size of each grid point [same on x and y], dt=1/samplerate
    gamma[0] = config['numerical_model_parameters'].getfloat('gamma') # type of edge, 0 if clamped edge, 1 if free edge
    
    w = config['numerical_model_parameters'].getint('w') # width of grid
    h = config['numerical_model_parameters'].getint('h') # height of grid
    dt = 1.0/config['numerical_model_parameters'].getfloat('samplerate') #uses samplerate from ini file to calculate.
    nsteps = config['numerical_model_parameters'].getint('nsteps')#number of steps
    
    #inital condition
    ex_input_freq[0] = config['numerical_model_parameters'].getint('bin') # bin determines frequency
    ex_input_mag[0] = config['numerical_model_parameters'].getfloat('magnitude') # amplitude of the wave
    ex_input_phase[0] =  config['numerical_model_parameters'].getfloat('phase') # phase of the wave (normalized 2π radians)

    if ex_input_freq[0] < 0:
        ex_input_freq[0] = 0
        print(f'{modelName}: Requested bin outside range, will be clipped to {ex_input_freq[0]}')   
    elif ex_input_freq[0] > (w-2)//2:
        ex_input_freq[0] = (w-2)//2
        print(f'{modelName}: Requested bin outside range, will be clipped to {ex_input_freq[0]}')
        
    if ex_input_phase[0] < 0:
        ex_input_phase[0] = 0.0
        print(f'{modelName}: Requested phase outside range [0, 1) (normalized 2π radians), will be clipped to {ex_input_phase[0]:.2f}')
    elif ex_input_phase[0] >= 1:
        ex_input_phase[0] = 0.99
        print(f'{modelName}: Requested phase outside range [0, 1) (normalized 2π radians), will be clipped to {ex_input_phase[0]:.2f}')

    #--------------------------------------------------------------------------------------

    # load
    _load(solver_path, prj_root, config_path)  # loads solver
    
    return


def _load(solver_path, prj_root, config_path):
    global solver
    solver, temp_var = import_file(prj_root, config_path, solver_path)

    return

def run(dev, b, dt, nsteps, w, h, mu, rho, gamma, ex_input_freq, ex_input_mag, ex_input_phase, disp=False, dispRate=1, pause=0):
    #function will be called by generator, all params passed at runtime (does not use global variables)
    #The arguments mu, rho, gamma, ex_x, ex_y, ex_amp are all arrays with b elements.

    # set parameters

    # propagation params, explained in solver
    # potentially model can have different values across domain (for now, model params only differ by batch, not across domain)
    _mu = torch.ones(b, h, w) 
    _rho = torch.ones(b, h, w)
    _gamma = torch.ones(b, h, w)
    
    # element-wise multiplication of tensor slice and vector
    # adapted from here: https://discuss.pytorch.org/t/element-wise-multiplication-of-a-vector-and-a-matrix/56946
    _mu = mu.view(-1, 1, 1).expand_as(_mu) * _mu
    _rho = rho.view(-1, 1, 1).expand_as(_rho) * _rho
    _gamma = gamma.view(-1, 1, 1).expand_as(_gamma) * _gamma
    
    #--------------------------------------------------------------
    # initial condition
    
    W = w-2 # -2 to remove boundary frame
    H = h-2 # -2 to remove boundary frame
    
    # number of bins with positive frequencies in both dimensions
    num_xFreqs = W//2 + 1 
    num_yFreqs = H//2 + 1
    
    xi0 = torch.zeros(b, H, W) # this will be overwritten with initial condition
    # create empty spatial frequency tensor, to initialize
    real = torch.zeros(b, H, num_xFreqs)
    imag = torch.zeros(b, H, num_xFreqs)
    freq = torch.complex(real, imag)
    # an alternative solution to switch to frequency domain:
    #freq = torch.fft.rfft2(xi0) # ready to initialize the domain in the spatial frequency domain!

    # spatial frequency initialization
    # polar turns polar coords to cartesian coords, or abs and angle to real and imaginary part
    freq[range(freq.shape[0]), 0, ex_input_freq] = torch.polar(ex_input_mag, 2*math.pi*ex_input_phase) 
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
    excite[..., 0] = xi0[...]
    # tensor indexing with range() instead of ':' adapted from here: https://stackoverflow.com/questions/61096522/pytorch-tensor-advanced-indexing

    #--------------------------------------------------------------
    # run solver
    sol, sol_t = solver.run(dev, dt, nsteps, b, w, h, _mu, _rho, _gamma, excite, torch.empty(0, 1), disp, dispRate, pause)

    return [sol, sol_t]

def run_test(dev, dispRate=1, pause=0):
    # set parameters
    _b = 1
    _disp = True
    
    
    #call run using those parameters+global variables, and return the result.
    test_sol, test_sol_t = run(dev, _b, dt, nsteps, w, h, mu, rho, gamma, ex_input_freq, ex_input_mag, ex_input_phase, _disp, dispRate, pause)
    
    return [test_sol, test_sol_t]


def getSolverInfo():
    if solver == 0:
        print(f'{modelName}: Cannot get description of solver! Model needs to be run at least once')
        return {'description':  ''}
    return solver.getInfo()
    
