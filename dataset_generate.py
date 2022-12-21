import torch
import configparser # to save config in new ini file
import scipy.io # to save dataset
from pathlib import Path # to properly handle paths and folders on every os
from timeit import default_timer # to measure processing time
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.utils import openConfig
from neuralacoustics.utils import import_fromScript
from neuralacoustics.utils import create_dataset_folder


# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)

#-------------------------------------------------------------------------------
# simulation parameters

# get config file + config path
config, config_path = getConfigParser(prj_root, __file__) # we call this script from command line directly
# hence __file__ is not a path, just the file name with extension

# read params from config file

# path to the generator folder
generator_path = config['dataset_generation'].get('dataset_generator')

# generator config file
generator_config_path = config['dataset_generation'].get('generator_config')
    
# chunks
ch = config['dataset_generation'].getint('chunks') # num of chunks

# dataset dir
dataset_dir = config['dataset_generation'].get('dataset_dir')  # where to save datasets

# device
dev_ = config['dataset_generation'].get('dev')  # cpu or gpu, keep original for dataset config
dev = dev_

# for quick visualization
dryrun = config['dataset_generation'].getint('dryrun') # visualize a single simulation run or save full dataset

# seconds to pause between datapoints during visualization
pause = config['dataset_generation'].getfloat('pause_sec')
# only used in dry run and it will be ignored in solver if <= 0. If -1, user input advances visualization

#-------------------------------------------------------------------------------------

# in case of generic gpu or cuda explicitly, check if available
if dev == 'gpu' or 'cuda' in dev:
  if torch.cuda.is_available():  
    dev = torch.device('cuda')
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
  else:  
    dev = torch.device('cpu')
    print('dataset_generator: gpu not available!')

print('Device:', dev)

#------------------------------------------------------------------------------
# import + load generator, extract parameters

generator_function_list = ['load, generate_datasetBatch, getSolverInfoFromModel']  # specify which functions to load.
generator, generator_config_path = import_fromScript(prj_root, config_path, generator_path, generator_config_path, function_list=generator_function_list)

num_of_batches, ch, N, B, h, w, nsteps, dt, model_config_path = generator.load(generator_config_path, ch, prj_root, pause) #return number of batches, chunks, remainder, after loading

batches_per_ch = num_of_batches//ch
ch_size = batches_per_ch * B  # num of data points per chunk

if dryrun == 0:

    # compute name of dataset + create folder
    dataset_folder, dataset_name = create_dataset_folder(prj_root, dataset_dir)
        
    #----------------------------------------------------------------------------
    # compute number of leading zeros for pretty file names
    ch_num = str(ch)
    l_zeros = len(ch_num) # number of leading zeros in file name
    # check if l_zeros needs to be lowered down
    # e.g., ch_num = 100 -> [0, 99] -> should be printed with only one leading zero:
    # 01, 02, ..., 98, 99
    cc = pow(10, l_zeros-1)
    if ch <= cc:
        l_zeros = l_zeros-1
    
    #-------------------------------------------------------------------------------
        
    time_duration = nsteps * dt
    print('simulation duration: ', time_duration, 's')
    
    # initial conditions (inputs) a 
    excite = torch.zeros(ch_size, h-2, w-2, nsteps+1) #VIC we will re-introduce at a certain point, to save continous excitation and other parameters, like mu and boundaries [first static then dynamic]
    
    # solutions u 
    sol = torch.zeros(ch_size, h, w, nsteps+1)  # +1 becase initial condition is saved at beginning of solution time series!

    ch_cnt = 0  # keeps track of # of chunks, datapoints during loop.
    n_cnt = 0
    rem = 0  # is there any remainder?
    t1 = default_timer()
    for b in range(num_of_batches):

        # compute all steps in full batch
        full_excitation, sol_b = generator.generate_datasetBatch(dev, dryrun) #generates dataset.

        # store
        excite[n_cnt:(n_cnt+B),...] = full_excitation # inputs
        sol[n_cnt:(n_cnt+B),...] = sol_b # results

        n_cnt += B

        # save some chunk, just in case...
        if (b+1) % batches_per_ch == 0: 
          file_name = dataset_name + '_ch' + str(ch_cnt).zfill(l_zeros) + '_' + str(n_cnt) + '.mat'
          dataset_path = dataset_folder.joinpath(file_name)
          scipy.io.savemat(dataset_path, mdict={'excite': excite.cpu().numpy(), 'sol': sol.cpu().numpy()})
          #scipy.io.savemat(dataset_path, mdict={'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
          print( '\tchunk {}, {} dataset points  (up to batch {} of {})'.format(ch_cnt, ch_size, b+1, num_of_batches) )
          ch_cnt += 1

          # reset initial conditions, solutions and data point count
          excite = torch.zeros(ch_size, h-2, w-2, nsteps+1)
          sol = torch.zeros(ch_size, h, w, nsteps+1) 

          n_cnt = 0
          
        elif (b+1) == num_of_batches:
          file_name = dataset_name + '_rem_' + str(n_cnt)  + '.mat'
          dataset_path = dataset_folder.joinpath(file_name)
          scipy.io.savemat(dataset_path, mdict={'excite': excite.cpu().numpy(), 'sol': sol.cpu().numpy()})
          #scipy.io.savemat(dataset_path, mdict={'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
          print( '\tremainder, {} dataset points (up to batch {} of {})'.format(n_cnt, b+1, num_of_batches) )
          rem = 1

    t2 = default_timer()

    rem_size = rem * n_cnt
    actual_size = (ch_size * ch) + rem_size

    print(f'\nDataset {dataset_name} saved in:')
    print('\t', dataset_folder)
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

    # save relevant bits from general config file + extra info from model config file into a new dataset config file (log)

    # create empty config file
    config = configparser.RawConfigParser()
    config.optionxform = str  # otherwise raw config parser converts all entries to lower case letters

    # fill it with dataset details
    config.add_section('dataset_details')
    config.set('dataset_details', 'name', dataset_name)
    config.set('dataset_details', 'actual_size', actual_size)
    config.set('dataset_details', 'simulated_time_s', time_duration)
    config.set('dataset_details', 'simulation_duration_s', simulation_duration)
    config.set('dataset_details', 'chunk_size', ch_size)
    config.set('dataset_details', 'remainder_size', rem_size)
    
    
    config.add_section('dataset_generation')
    config.set('dataset_generation', 'dataset_generator', generator_path)
    config.set('dataset_generation', 'dev', dev_)
    config.set('dataset_generation', 'chunks', ch)
    config.set('dataset_generation', 'dataset_dir', dataset_dir)
    config.set('dataset_generation', 'dryrun', dryrun)

    # the model config file is the current config file
    # by doing so, dataset too can be re-built using this config file
    config.set('dataset_generation', 'generator_config', 'this_file')

    # then retrieve generator and solver details from generator config file
    config_gen = openConfig(generator_config_path, __file__)

    # extract relevant bits and add them to new dataset config file
    config.add_section('dataset_generator_details')
    for(each_key, each_val) in config_gen.items('dataset_generator_details'):
        config.set('dataset_generator_details', each_key, each_val)
    
    config.add_section('generator_params_details')
    for(each_key, each_val) in config_gen.items('generator_params_details'):
        config.set('generator_params_details', each_key, each_val)
        
    config.add_section('dataset_generator_parameters')
    for(each_key, each_val) in config_gen.items('dataset_generator_parameters'):
        config.set('dataset_generator_parameters', each_key, each_val)
    config.set('dataset_generator_parameters', 'numerical_model_config', 'this_file')

    config.add_section('numerical_model_parameters')
    for(each_key, each_val) in config_gen.items('numerical_model_parameters'):
        config.set('numerical_model_parameters', each_key, each_val)        
    
    #retrieve model and solver details from model config file
    config_model = openConfig(model_config_path, __file__)
    
    config.add_section('numerical_model_details')
    for(each_key, each_val) in config_model.items('numerical_model_details'):
        config.set('numerical_model_details', each_key, each_val)
   
    config.add_section('solver')
    for(each_key, each_val) in config_model.items('solver'): #solver name & directory from model config file
        config.set('solver', each_key, each_val)
    for(each_key, each_val) in generator.getSolverInfoFromModel().items(): #solver description from generator
        config.set('solver', each_key, each_val)

    # where to write it
    dataset_config_path = dataset_folder.joinpath(dataset_name+'.ini') 
    # write
    with open(dataset_config_path, 'w') as configfile:
        config.write(configfile)

else:
    generator.generate_datasetBatch(dev, dryrun) #generate 1 batch with display on.
