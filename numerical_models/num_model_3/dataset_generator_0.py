import torch
import configparser # to save config in new ini file
import scipy.io # to save dataset
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time
from neuralacoustics.utils import openConfig

numerical_model = ''
numerical_model_dir = ''
dataset_root = ''
solver_dir = ''
solver_name = ''
N = -1
B = -1
w = -1
h = -1
seed = -1
nsteps = -1
dt = -1
ch = -1
pause_sec = -1
model = 0

def load_generator(config_path, prj_root):
    # in same style as load_test, this function writes the config variables to global variables.
    global numerical_model 
    global numerical_model_dir 
    global solver_dir
    global solver_name 
    global N 
    global B 
    global w 
    global h
    global nsteps 
    global dt
    global dryrun 
    global pause_sec
    global model
    global dataset_root
    global ch
    
    #open config file
    generator_name = Path(__file__).stem
    config = openConfig(config_path, generator_name)
    
    #------------------------------------------------------------------------------------------------------------------------------------
    #model
    
    numerical_model = config['dataset_generation'].get('numerical_model') #name of num_model
    numerical_model_dir = config['dataset_generation'].get('numerical_model_dir') #directory of num_model
    
    model_path_folders = Path(numerical_model_dir.replace('PRJ_ROOT', '.')).joinpath(numerical_model).parts
    # ^ also add folder with same name as model (so we have ./{numerical_model_dir}/{numerical_model}/)

    # create package structure by concatenating folders with '.'
    packages_struct = model_path_folders[0]
    for pkg in range(1,len(model_path_folders)):
        packages_struct += '.'+model_path_folders[pkg] 
    
    # load 
    model = __import__(packages_struct + '.' + numerical_model, fromlist=['*']) # model.path.numerical_model is model script [i.e., package]
    
    #-----------------------------------------------------------------------------------------------------------------------------------
       #parameters
    
    # dataset size
    N = config['dataset_generation'].getint('N') # num of dataset points
    B = config['dataset_generation'].getint('B') # batch size
    
    # domain size
    w = config['dataset_generation'].getint('w') # domain width [cells]
    h = config['dataset_generation'].getint('h') # domain height[cells]
    
    # time parameters
    nsteps = config['dataset_generation'].getint('nsteps') # = T_in+T_out, e.g., Tin = 10, T = 10 -> input [0,Tin), output [10, Tin+T)
    dt = 1.0 / config['dataset_generation'].getint('samplerate') # seconds (1/Hz), probably no need to ever modify this...
    
    # chunks
    ch = config['dataset_generation'].getint('chunks') # num of chunks
    
    # dataset dir
    dataset_root = config['dataset_generation'].get('dataset_root') #where to save files.
    dataset_root = dataset_root.replace('PRJ_ROOT', prj_root)
    # ^ could replace this with '.' and it'll still work, not sure if it'll break something else down the line tho
    
    #for quick visualization
    dryrun = config['dataset_generation'].getint('dryrun') # visualize a single simulation run or save full dataset
    # seconds to pause between datapoints during visualization
    pause = config['dataset_generation'].getfloat('pause_sec')
    # only used in dry run and it will be ignored in solver if <= 0
    
    #solver parameters (used for model.load)
    solver_dir = config['solver'].get('solver_dir')
    solver_name = config['solver'].get('solver_name')
    
    return


def generate_dataset(dev):
    global ch
    global N
    
    #dryrun off (should dryrun just be a bool?)
    if dryrun == 0:
        # compute name of dataset + create folder
        #----------------------------------------------------------------------------
        # count datasets in folder
        datasets = list(Path(dataset_root).glob('*'))
        num_of_datasets = len(datasets)
        # choose new dataset index accordingly
        DATASET_INDEX = str(num_of_datasets)

        name_clash = True

        while name_clash:
            name_clash = False
            for dataset in datasets:
                # in case a dataset with same name is there
                if Path(dataset).parts[-1] == 'dataset_'+DATASET_INDEX:
                    name_clash = True
                    DATASET_INDEX = str(int(DATASET_INDEX)+1) # increase index

        dataset_name = 'dataset_' + DATASET_INDEX
        dataset_dir = Path(dataset_root).joinpath(dataset_name)

        # create folder where to save dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        #----------------------------------------------------------------------------
        # load model and compute meta data, e.g., duration, actual size...

        nn = model.load(solver_dir, solver_name)
        n_zeros = 0
        # if this is a grid model...
        if nn != None:
          # ...we ignore the requested data point num and use the one built into the model [i.e., number of grid inputs]
          if nn%B > 0:
            # if num of data poins is not an integer multiple of batch size
            N = B * (1 + nn//B) # we round it up to the closest multiple of batch size, because this is what the model will do, 
            # it will add silence for all the simulations that exceed the number of grid inputs defined in the model's ini file
            n_zeros = N-nn # we save the number of zero data points that are addeded at the end of dataset
          else:
            # otherwise, no need to round up
            N = nn
        
        time_duration = nsteps * dt

        print('simulation duration: ', time_duration, 's')

        # num of chunks must be lower than total number of batches
        if ch > N//B:
          ch = (N//B)//2 # otherwise, a chunk every other batch
        if ch == 0: # always at least one chunk!
          ch = 1

          
        # it is possible that not all requested points are generated,
        # depending on ratio with requested batch size
        # this is why we calculate the actual_size of the dataset once saved
        # this is not true for grid-based models [they return have fixed numbers of points that takes batch size into account]
        n_cnt=0
        num_of_batches = N//B
        batches_per_ch = num_of_batches//ch
        ch_size = batches_per_ch * B # num of data points per chunk
        ch_cnt = 0
        rem = 0 # is there any remainder?

        # compute number of leading zeros for pretty file names
        ch_num = str(ch)
        l_zeros=len(ch_num) # number of leading zeros in file name
        # check if l_zeros needs to be lowered down
        # e.g., ch_num = 100 -> [0, 99] -> should be printed with only one leading zero:
        # 01, 02, ..., 98, 99
        cc = pow(10,l_zeros-1)
        if ch <= cc:
          l_zeros = l_zeros-1

        
        #----------------------------------------------------------------------------
        # compute full dataset and save it

        t1 = default_timer()

        # initial conditions
        ex_x, ex_y, ex_amp, mu, rho, gamma = generate_rand_tensors(w, h, B) #create rand tensors for excitation and medium
        #a = torch.zeros(ch_size, h, w) #VIC we will re-introduce at a certain point, to save continous excitation and other parameters, like mu and boundaries [first static then dynamic]
        
        # solutions
        u = torch.zeros(ch_size, h, w, nsteps+1) # +1 becase initial condition is saved at beginning of solution time series!


        for b in range(num_of_batches):
            # compute all steps in full batch
            sol, sol_t = model.run(dev, B, dt, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, ex_amp)
            
            # store
            u[n_cnt:(n_cnt+B),...] = sol # results

            n_cnt += B

            # save some chunk, just in case...
            if (b+1) % batches_per_ch == 0: 
              file_name = dataset_name + '_ch' + str(ch_cnt).zfill(l_zeros) + '_' + str(n_cnt) + '.mat'
              dataset_path = dataset_dir.joinpath(file_name)
              #scipy.io.savemat(dataset_path, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
              scipy.io.savemat(dataset_path, mdict={'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
              print( '\tchunk {}, {} dataset points  (up to batch {} of {})'.format(ch_cnt, ch_size, b+1, num_of_batches) )
              ch_cnt += 1
              # reset initial conditions, solutions and data point count
              u = torch.zeros(ch_size, h, w, nsteps+1) # +1 becase initial condition is repeated at beginning of solution time series!
              n_cnt = 0
            elif (b+1) == num_of_batches:
              file_name = dataset_name + '_rem_' + str(n_cnt)  + '.mat'
              dataset_path = dataset_dir.joinpath(file_name)
              scipy.io.savemat(dataset_path, mdict={'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
              print( '\tremainder, {} dataset points (up to batch {} of {})'.format(n_cnt, b+1, num_of_batches) )
              rem = 1

        t2 = default_timer()

        rem_size = rem*n_cnt
        actual_size = (ch_size * ch) + rem_size

        print(f'\nDataset {dataset_name} saved in:')
        print('\t', dataset_dir)
        print(f'total number of data points: {actual_size} (out of {N} requested)')
        if ch == 1:
            print('in a single chunk')
        else:
            print(f'split in {ch_cnt} chunks with {ch_size} datapoints each')
            if rem:
              print(f'plus remainder file with {n_cnt} datapoints')

        simulation_duration = t2-t1
        print(f'\nElapsed time: {simulation_duration} s\n')
    
    # ------------------------------------------------------------------------------------------------
    #dryrun on
    else:
        ex_x, ex_y, ex_amp, mu, rho, gamma = generate_rand_tensors(w, h, 1) #create rand tensors for excitation and medium
        model.load(solver_dir, solver_name) #loads solver for model
        model.run(dev, 1, dt, nsteps, w, h, mu, rho, gamma, ex_x, ex_y, ex_amp, disp=True, dispRate=1, pause=pause_sec)
        #run model (B = 1 entry) with display on
        
    return


def generate_rand_tensors(_w, _h, _B, force_det = False, _seed = 0):
    if force_det:
        torch.manual_seed(_seed) #used to seed tensors.
    rd_x = torch.randint(0, _w-2, (_B,)) 
    rd_y = torch.randint(0, _h-2, (_B,))
    rd_amp = torch.randn(_B)
    rd_mu = torch.rand(_B) #these will prob need adjustment in order to match realistic values.
    rd_rho = torch.rand(_B)
    rd_gamma = torch.rand(_B)
    return rd_x, rd_y, rd_amp, rd_mu, rd_rho, rd_gamma
