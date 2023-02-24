import torch
from timeit import default_timer
import configparser
from pathlib import Path
from datetime import datetime # for current date time in name of model
import socket # for hostname in name of model

# to load dataset
from neuralacoustics.DatasetManager import DatasetManager
from neuralacoustics.utils import LpLoss
from neuralacoustics.utils import seed_worker # for PyTorch DataLoader determinism
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.utils import openConfig
from neuralacoustics.utils import count_params
from neuralacoustics.utils import UnitGaussianNormalizer
from neuralacoustics.adam import Adam # adam implementation that deals with complex tensors correctly [lacking in pytorch <=1.8, not sure afterwards]
from torch.utils.tensorboard import SummaryWriter

# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)



#-------------------------------------------------------------------------------
# training parameters

# get config file
config, _ = getConfigParser(prj_root, __file__) # we call this script from command line directly
# hence __file__ is not a path, just the file name with extension

# read params from config file

# dataset parameters
# dataset name
dataset_name = config['training'].get('dataset_name')
# dataset dir
dataset_dir_ = config['training'].get('dataset_dir') # keep original string for log
dataset_dir = dataset_dir_.replace('PRJ_ROOT', prj_root)

# number of data points to load, i.e., specific sub-series of time steps within data entries
n_train = config['training'].getint('n_train')
n_test = config['training'].getint('n_test')

# input and output steps
T_in = config['training'].getint('T_in') 
T_out = config['training'].getint('T_out')
# T_in+T_out is window size!

# offset between consecutive windows
win_stride = config['training'].getint('window_stride') 

# maximum index of the frame (timestep) that can be retrieved from each dataset entry
win_limit = config['training'].getint('window_limit') 

# permute dataset entries
permute = config['training'].getint('permute') 
permute = bool(permute>0)

# network
network_name = config['training'].get('network_name')
#network_dir_ = config['training'].get('network_dir')
#network_dir = Path(network_dir_.replace('PRJ_ROOT', prj_root)) / network_name
#network_path = network_dir / (network_name + '.py')
network_root = Path( config['training'].get('network_dir').replace('PRJ_ROOT', prj_root) )
network_dir = network_root.joinpath(network_name)
network_path = network_dir.joinpath(network_name+'.py')

network_config_path = config['training'].get('network_config')
if network_config_path == 'default' or network_config_path == '':
    #network_config_path = network_dir / (network_name + '.ini')
    network_config_path = network_dir.joinpath(network_name + '.ini')
else:
    network_config_path = Path(network_config_path.replace('PRJ_ROOT', prj_root))

# Store a config instance for later logging use
network_config = openConfig(network_config_path, __file__)

# Read network inferrence type
inference_type = network_config['network_details'].get('inference_type')

# Load network
# we want to load the package through potential subfolders
# we can pretend we are in the PRJ_ROOT, for __import__ will look for the package from there
network_path_folders = network_path.parts
# create package structure by concatenating folders with '.'
network_path_struct = '.'.join(network_path_folders)[:-3] # append all parts and remove '.py' from file/package name
network_mod = __import__(network_path_struct, fromlist=['*'])
network = getattr(network_mod, network_name)

# training parameters
batch_size = config['training'].getint('batch_size')

if batch_size>n_train or n_train%batch_size!=0:
    print(f'##### Warning: batch size {batch_size} not submultiple of number of training points {n_train}')
if batch_size>n_test or n_test%batch_size!=0:
    print(f'##### Warning: batch size {batch_size} not submultiple of number of test points {n_test}')


epochs = config['training'].getint('epochs')

learning_rate = config['training'].getfloat('learning_rate')
scheduler_step = config['training'].getint('scheduler_step')
scheduler_gamma = config['training'].getfloat('scheduler_gamma')

checkpoint_step = config['training'].getint('checkpoint_step')

# Normalization
normalize = config['training'].getint('normalize_data')


# misc parameters
model_root_ = config['training'].get('model_dir') # keep original for config file
model_root = model_root_.replace('PRJ_ROOT', prj_root)
model_root = Path(model_root)

load_model_name = config['training'].get('load_model_name')
load_model_checkpoint = config['training'].get('load_model_checkpoint')

seed = config['training'].getint('seed')

dev = config['training'].get('dev')

print('Model and training parameters:')
print(f'\tdataset name: {dataset_name}')
print(f'\trequested training data points: {n_train}')
print(f'\trequested training test points: {n_test}')
print(f'\tinput steps: {T_in}')
print(f'\toutput steps: {T_out}')
# print(f'\tmodes: {modes}')
# print(f'\twidth: {width}')
print(f'\tbatch size: {batch_size}')
print(f'\tepochs: {epochs}')
print(f'\tlearning_rate: {learning_rate}')
print(f'\tscheduler_step: {scheduler_step}')
print(f'\tscheduler_gamma: {scheduler_gamma}')
print(f'\tcheckpoint_step: {checkpoint_step}')
print(f'\trandom seed: {seed}')


#-------------------------------------------------------------------------------
# determinism
# https://pytorch.org/docs/stable/notes/randomness.html
# generic
torch.use_deterministic_algorithms(True) 
torch.backends.cudnn.deterministic = True 

# needed for DataLoader 
torch.manual_seed(seed) # for permutation in loadDataset() and  seed_worker() in utils.py
g = torch.Generator()
g.manual_seed(seed)

#-------------------------------------------------------------------------------
# dirs, paths, files

# Determine the checkpoint path if a loaded model is specified
load_model_path = None
if load_model_name != "":
    load_model_path = Path(model_root).joinpath(load_model_name).joinpath(
        'checkpoints').joinpath(load_model_checkpoint)

    # Use the last checkpoint if the provided checkpoint is not valid
    if not load_model_path.is_file():
        load_checkpoint_path = Path(model_root).joinpath(
            load_model_name).joinpath('checkpoints')
        checkpoints = [x.name for x in list(load_checkpoint_path.glob('*'))]
        
        if len(checkpoints) < 1:
            raise FileNotFoundError("No available checkpoint")

        checkpoints.sort()
        load_model_checkpoint = checkpoints[-1]
        load_model_path = load_checkpoint_path.joinpath(load_model_checkpoint)
    
    print()
    print(f"Load from model: {load_model_name}")
    print(f"Load from checkpoint: {load_model_checkpoint}")

# Determine saved model name and directory
if load_model_name != "":
    model_name = load_model_name + datetime.now().strftime('_%y-%m-%d_%H-%M_continued')
else:
    # date time and local host name
    model_name = datetime.now().strftime('%y-%m-%d_%H-%M_'+socket.gethostname())

model_dir = model_root.joinpath(model_name) # the directory contains an extra folder with same name of model, that will include both model and log file
    
# create folder where to save model and log file
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir.joinpath(model_name) # full path to model: dir+name

# Create model checkpoint folder
model_checkpoint_dir = model_dir.joinpath("checkpoints")
model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Open txt file for logging
f = open(str(model_path)+'.log', 'w')

# tensorboard
writer = SummaryWriter(str(model_dir.joinpath("tensorboard")))
# to view: 
# from tensordboard model dir
# run: tensorboard --logdir=. 
# then open/click on http://localhost:6006/


#-------------------------------------------------------------------------------
# retrieve all data points
print() # a new line   

t1 = default_timer()

data_manager = DatasetManager(dataset_name, dataset_dir)
u = data_manager.loadData(
    n=n_train+n_test,
    win=T_in+T_out,
    stride=win_stride,
    win_lim=win_limit,
    start_ch=0,
    permute=permute
)

# get domain size
sh = list(u.shape)
S = sh[1] 
# we assume that all datasets have simulations spanning square domains
assert(S == sh[2])

# prepare train and test sets 
train_a = u[:n_train,:,:,:T_in]
train_u = u[:n_train,:,:,T_in:T_in+T_out]
test_a = u[-n_test:,:,:,:T_in]
test_u = u[-n_test:,:,:,T_in:T_in+T_out]

a_normalizer = None
y_normalizer = None
if normalize:
    print("Normalizing input and output data...")
    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

#print(train_u.shape, test_u.shape)
assert(S == train_u.shape[-2])
assert(T_out == train_u.shape[-1])

# TODO: incorporate different input data form
# train_a = train_a.reshape(n_train,S,S,T_in)
# test_a = test_a.reshape(n_test,S,S,T_in)
train_a = train_a.reshape(n_train,S,S,1,T_in).repeat([1,1,1,40,1])
test_a = test_a.reshape(n_test,S,S,1,T_in).repeat([1,1,1,40,1])

num_workers = 1 # for now single-process data loading, called explicitly to assure determinism in future multi-process calls

# datapoints will be loaded from these
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True, 
num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False, 
num_workers=num_workers, worker_init_fn=seed_worker, generator=g) #VIC not sure if seed_worker and generator needed here, even in multi-process calls
# because there is no shuffle

t2 = default_timer()

print(f'\nDataset preprocessing finished, elapsed time: {t2-t1} s')
print(f'Training input shape: {train_a.shape}, output shape: {train_u.shape}')    



#-------------------------------------------------------------------------------
# select device and create model
print(f'\nModel name: {model_name}')


# in case of generic gpu or cuda explicitly, check if available
if dev == 'gpu' or 'cuda' in dev:
    if torch.cuda.is_available():
        model = network(network_config_path, T_in).cuda()
        dev = torch.device('cuda')
        if normalize:
            y_normalizer.cuda()
        #print(torch.cuda.current_device())
        #print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('GPU/Cuda not available, switching to CPU...')
        model = network(network_config_path, T_in)
        dev  = torch.device('cpu')
        if normalize:
            y_normalizer.cpu()
else:
    model = network(network_config_path, T_in)
    dev  = torch.device('cpu')
    if normalize:
        y_normalizer.cpu()

print('Device:', dev)


print(f'Number of model\'s parameters: {count_params(model)}')
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9) # this would need to be modified to handle complex arithmetic
# TODO: incorporate different scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
iterations = epochs * (n_train // batch_size)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

# Load previous checkpoint
prev_ep = 0
if load_model_name != "":
    state_dict = torch.load(load_model_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    scheduler.load_state_dict(state_dict['scheduler_state_dict'])
    prev_ep = state_dict['epoch'] + 1
#-------------------------------------------------------------------------------
# train!

print('\n___Start training!___')
t1 = default_timer()

# log and print headers
# not using same string due to formatting visualization differences
log_str = 'Epoch\tDuration\t\t\t\tLoss Step Train\t\t\tLoss Full Train\t\t\tLoss Step Test\t\t\tLoss Full Test'
f.write(log_str)
print('Epoch\tDuration\t\t\tLoss Step Train\t\t\tLoss Full Train\t\t\tLoss Step Test\t\t\tLoss Full Test')

myloss = LpLoss(size_average=False)
for ep in range(epochs):

    #--------------------------------------------------------
    # train
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    # for xx, yy in train_loader:
    #     loss = 0
    #     xx = xx.to(dev)
    #     yy = yy.to(dev)

    #     # model outputs 1 timestep at a time [i.e., labels], so we iterate over T_out steps to compute loss
    #     for t in range(0, T_out):
    #         y = yy[..., t:t+1]
    #         im = model(xx)
    #         loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

    #         if t == 0:
    #             pred = im
    #         else:
    #             pred = torch.cat((pred, im), -1)

    #         xx = torch.cat((xx[..., 1:], im), dim=-1)

    #     train_l2_step += loss.item()
    #     l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
    #     train_l2_full += l2_full.item()
    #     #VIC not sure why not simply train_l2_full += myloss(...) and get rid of l2_full at once [as in test], but the result is slightly different!!!

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(dev)
        yy = yy.to(dev)

        optimizer.zero_grad()
        pred = model(xx)
        pred = pred.view(batch_size, S, S, 40)

        if normalize:
            pred = y_normalizer.decode(pred)
            yy = y_normalizer.decode(yy)

        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        l2_full.backward()
        optimizer.step()
        scheduler.step()

        train_l2_full += l2_full.item()
        

    #--------------------------------------------------------
    # test
    test_l2_step = 0
    test_l2_full = 0
    model.eval()
    # with torch.no_grad():
    #     for xx, yy in test_loader:
    #         loss = 0
    #         xx = xx.to(dev)
    #         yy = yy.to(dev)

    #         for t in range(0, T_out):
    #             y = yy[..., t:t+1]
    #             im = model(xx)
    #             loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

    #             if t == 0:
    #                 pred = im
    #             else:
    #                 pred = torch.cat((pred, im), -1)

    #             xx = torch.cat((xx[..., 1:], im), dim=-1)

    #         test_l2_step += loss.item()
    #         test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
    
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(dev)
            yy = yy.to(dev)

            pred = model(xx).view(batch_size, S, S, 40)
            if normalize:
                pred = y_normalizer.decode(pred)
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()


    #--------------------------------------------------------
    #log

    # tensorboard log
    epoch_train_loss_step =  train_l2_step / n_train / T_out
    epoch_train_loss_full =  train_l2_full / n_train

    epoch_test_loss_step =  test_l2_step / n_test / T_out
    epoch_test_loss_full =  test_l2_full / n_test

    writer.add_scalar("Loss Step/train", epoch_train_loss_step, ep + prev_ep)
    writer.add_scalar("Loss Full/train", epoch_train_loss_full, ep + prev_ep)

    writer.add_scalar("Loss Step/test", epoch_test_loss_step, ep + prev_ep)
    writer.add_scalar("Loss Full/test", epoch_test_loss_full, ep + prev_ep)

    # log file and print
    # not using same string due to formatting visualization differences
    f.write('\n')
    log_str = '{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(ep + prev_ep, t2-t1, epoch_train_loss_step, epoch_train_loss_full, epoch_test_loss_step, epoch_test_loss_full)
    f.write(log_str)
    print(f'{ep + prev_ep}\t{t2 - t1}\t\t{epoch_train_loss_step}\t\t{epoch_train_loss_full}\t\t{epoch_test_loss_step}\t\t{epoch_test_loss_full}')


    #--------------------------------------------------------
    # Save model, optimizer and scheduler status every checkpoint_step epochs
    if checkpoint_step >= 1 and (ep + 1) % checkpoint_step == 0:
        save_model_name = model_name + '_ep{:04d}'.format(ep + prev_ep) + '.pt'
        save_model_path = model_checkpoint_dir.joinpath(save_model_name)
        torch.save({
            'epoch': ep + prev_ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'a_normalizer': a_normalizer,
            'y_normalizer': y_normalizer,
        },
        save_model_path)
        print(f"\t----> checkpoint {save_model_name} saved")


f.close()
#writer.flush()
writer.close()

# final loss with 4 decimals    
final_train_loss = '{:.4f}'.format(epoch_train_loss_full)
final_test_loss = '{:.4f}'.format(epoch_test_loss_full)

print('___Training done!___')
t2 = default_timer()
train_duration = t2-t1
print(f"Elapsed time: {train_duration} s")


#-------------------------------------------------------------------------------
# Save the final model checkpoint, only when it hasn't been saved yet
if checkpoint_step < 1 or (checkpoint_step >= 1 and epochs % checkpoint_step != 0):
    save_model_name = model_name + '_ep{:04d}'.format(epochs-1+prev_ep) + '.pt'
    save_model_path = model_checkpoint_dir.joinpath(save_model_name)
    torch.save({
        'epoch': epochs - 1 + prev_ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'a_normalizer': a_normalizer,
        'y_normalizer': y_normalizer,
    },
    save_model_path)
    print(f"----> Final checkpoint {save_model_name} saved")


# save model to folder
torch.save(model, model_path)

print(f'\nModel {model_name} saved in:')
print('\t', model_dir)

print(f'final train loss: {final_train_loss}')
print(f'final test loss: {final_test_loss}\n')


#-------------------------------------------------------------------------------
# save model info + training info from used config file into a new model config file (log)

# create empty config file for model 
config_model = configparser.RawConfigParser()
config_model.optionxform = str # otherwise raw config parser converts all entries to lower case letters



# fill it with model details
config_model.add_section('model_details')
config_model.set('model_details', 'name', model_name)
config_model.set('model_details', 'train_loss', final_train_loss)
config_model.set('model_details', 'test_loss', final_test_loss)
config_model.set('model_details', 'training_duration', train_duration)



# then training details, from config file used
config_model.add_section('training')
for(each_key, each_val) in config.items('training'):
      config_model.set('training', each_key, each_val)

# Add network detail
config_model.add_section('network_params_details')
config_model.add_section('network_parameters')
for (k, v) in network_config.items('network_params_details'):
    config_model.set('network_params_details', k, v)
for (k, v) in network_config.items('network_parameters'):
    config_model.set('network_parameters', k, v)

config_model.set('network_parameters', 'T_in', T_in)

# then retrieve all content of dataset config file
dataset_dir = Path(dataset_dir)
dataset_config_path = dataset_dir.joinpath(dataset_name).joinpath(dataset_name+'.ini') 

config = openConfig(dataset_config_path, __file__)

for(each_section,_) in config.items():
    if(each_section != 'DEFAULT'):
        config_model.add_section(each_section)
        for(each_key, each_val) in config.items(each_section):
            if(each_key != 'numerical_model_config'):
                config_model.set(each_section, each_key, each_val)
            else:
                # currrent config file [same name as model and in same dir], but expressed with original root, that may contain PRJ_ROOT     
                config_path = Path(model_root_).joinpath(model_name).joinpath(model_name +'.ini') # maintains PRJ_ROOT var [if used]
                config_path = str(config_path)
                config_model.set(each_section, each_key, config_path)
                # by doing so, dataset too can be re-built using this config file



# where to write it
config_path = model_dir.joinpath(model_name +'.ini') # same name as model and in same dir
# write
with open(config_path, 'w') as configfile:
    config_model.write(configfile)
