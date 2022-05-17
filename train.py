import torch
from timeit import default_timer
import configparser
from pathlib import Path

from neuralacoustics.model import FNO2d
from neuralacoustics.dataset_loader import loadDataset # to load dataset
from neuralacoustics.utils import LpLoss
from neuralacoustics.utils import seed_worker # for PyTorch DataLoader determinism
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.utils import count_params
from neuralacoustics.adam import Adam # adam implementation that deals with complex tensors correctly [lacking in pytorch <=1.8, not sure afterwards]

from torch.utils.tensorboard import SummaryWriter

# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)


#-------------------------------------------------------------------------------
# training parameters

# get config file
config = getConfigParser(prj_root, __file__.replace('.py', ''))

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
# by default, windows are juxtaposed
if win_stride <= 0:
    win_stride = T_in+T_out

# maximum index of the frame (timestep) that can be retrieved from each dataset entry
win_limit = config['training'].getint('window_limit') 


# network parameters
modes = config['training'].getint('network_modes')
width = config['training'].getint('network_width')


# training parameters
batch_size = config['training'].getint('batch_size')

epochs = config['training'].getint('epochs')

learning_rate = config['training'].getfloat('learning_rate')
scheduler_step = config['training'].getint('scheduler_step')
scheduler_gamma = config['training'].getfloat('scheduler_gamma')


# misc parameters
model_root = config['training'].get('model_dir')
model_root = model_root.replace('PRJ_ROOT', prj_root)
model_root = Path(model_root)

seed = config['training'].getint('seed')

dev = config['training'].get('dev')

print('Model and training parameters:')
print(f'\tdataset name: {dataset_name}')
print(f'\trequested training data points: {n_train}')
print(f'\trequested training test points: {n_test}')
print(f'\tinput steps: {T_in}')
print(f'\toutput steps: {T_out}')
print(f'\tmodes: {modes}')
print(f'\twidth: {width}')
print(f'\tepochs: {epochs}')
print(f'\tlearning_rate: {learning_rate}')
print(f'\tscheduler_step: {scheduler_step}')
print(f'\tscheduler_gamma: {scheduler_gamma}')
print(f'\trandom seed: {seed}')


#-------------------------------------------------------------------------------
# determinism
# https://pytorch.org/docs/stable/notes/randomness.html
torch.use_deterministic_algorithms(True) # generic
torch.backends.cudnn.deterministic = True #VIC probably this should be called only in those models that use it

# needed for DataLoader 
torch.manual_seed(seed) # check seed_worker() in utils.py
g = torch.Generator()
g.manual_seed(seed)

#-------------------------------------------------------------------------------
# compute name of model

# count datastes in folder 
models = list(Path(model_root).glob('*'))
num_of_models = len(models)
# choose new model index accordingly
MODEL_INDEX = str(num_of_models)

name_clash = True

while name_clash:
  name_clash = False
  for model in models:
    # in case a model with same name is there
    if Path(model).parts[-1] == 'model_'+MODEL_INDEX:
      name_clash = True
      MODEL_INDEX = str(int(MODEL_INDEX)+1) # increase index

model_name = 'model_'+MODEL_INDEX
model_dir = model_root.joinpath(model_name) # the directory contains an extra folder with same name of model, that will include both model and log file

# create folder where to save model and log file
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir.joinpath(model_name)

# log
# txt file
f = open(str(model_path)+'.log', 'w')
# tensorboard
writer = SummaryWriter(str(model_dir)+'/tensorboard')
# to view: 
# from tensordboard model dir
# run: tensorboard --logdir=. 
# then open/click on http://localhost:6006/


#-------------------------------------------------------------------------------
# retrieve all data points
print() # a new line   

t1 = default_timer()

u = loadDataset(dataset_name, dataset_dir, n_train+n_test, T_in+T_out, win_stride, win_limit)
# get domain size
sh = list(u.shape)
S = sh[1] 
# we assume that all datasets have simulations spanning square domains
assert(S == sh[2])

# prepare train and and test sets 
train_a = u[:n_train,:,:,:T_in]
train_u = u[:n_train,:,:,T_in:T_in+T_out]
test_a = u[-n_test:,:,:,:T_in]
test_u = u[-n_test:,:,:,T_in:T_in+T_out]

#print(train_u.shape, test_u.shape)
assert(S == train_u.shape[-2])
assert(T_out == train_u.shape[-1])

train_a = train_a.reshape(n_train,S,S,T_in)
test_a = test_a.reshape(n_test,S,S,T_in)

num_workers = 1 # for now single-process data loading, called explicitly to assure determinism in future multi-process calls

# datapoints will be loaded from these
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True, 
num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False, 
num_workers=num_workers, worker_init_fn=seed_worker, generator=g) #VIC not sure if seed_worker and generator needed here, even multi-process calls
# because there is no shuffle

t2 = default_timer()

print(f'\nDataset preprocessing finished, time used: {t2-t1}s')
print(f'Training input shape: {train_a.shape}, output shape: {train_u.shape}')    



#-------------------------------------------------------------------------------
# select device and create model
print(f'\nModel name: {model_name}')


# in case of generic gpu or cuda explicitly, check if available
if dev == 'gpu' or 'cuda' in dev:
  if torch.cuda.is_available():  
    model = FNO2d(modes, modes, width, T_in).cuda()
    dev  = torch.device('cuda')
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
  model = FNO2d(modes, modes, width, T_in)
  dev  = torch.device('cpu')

print('Device:', dev)


print(f'Nunmber of model\'s parameters: {count_params(model)}')
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


#-------------------------------------------------------------------------------
# train!

print('\n___Start training!___')
#VIC test
# train_features, train_labels = next(iter(train_loader))
# print(train_features.size())
# print(train_labels.size())
# t1 = default_timer()
# im = model(train_features)
# t2 = default_timer()
# print(f'\nInference finished, time used: {t2-t1}s')
# quit()

# log and print headers
# not using same string due to formatting errors
log_str = 'Epoch\tDuration\t\t\t\tLoss Step Train\t\t\tLoss Full Train\t\t\tLoss Step Test\t\t\tLoss Full Test'
f.write(log_str)
print('Epoch\tDuration\t\t\tLoss Step Train\t\t\tLoss Full Train\t\t\tLoss Step Test\t\t\tLoss Full Test')

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(dev)
        yy = yy.to(dev)

        # model outputs 1 timestep at a time [i.e., labels], so we iterate over T_out steps to compute loss
        for t in range(0, T_out):
            y = yy[..., t:t+1]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., 1:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()
        #VIC not sure why not simply train_l2_full += myloss(...) and get rid of l2_full at once [as in test], but the result is slightly different!!!

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    # test
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(dev)
            yy = yy.to(dev)

            for t in range(0, T_out):
                y = yy[..., t:t+1]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., 1:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()

    # tensorboard log
    epoch_train_loss_step =  train_l2_step / n_train / T_out
    epoch_train_loss_full =  train_l2_full / n_train

    epoch_test_loss_step =  test_l2_step / n_train / T_out
    epoch_test_loss_full =  test_l2_full / n_train

    writer.add_scalar("Loss Step/train", epoch_train_loss_step, ep)
    writer.add_scalar("Loss Full/train", epoch_train_loss_full, ep)

    writer.add_scalar("Loss Step/test", epoch_test_loss_step, ep)
    writer.add_scalar("Loss Full/test", epoch_test_loss_full, ep)

    # log and print
    # not using same string due to formatting errors
    f.write('\n')
    log_str = '{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(ep, t2-t1, epoch_train_loss_step, epoch_train_loss_full, epoch_test_loss_step, epoch_test_loss_full)
    f.write(log_str)
    print(f'{ep}\t{t2 - t1}\t\t{epoch_train_loss_step}\t\t{epoch_train_loss_full}\t\t{epoch_test_loss_step}\t\t{epoch_test_loss_full}')

f.close()
#writer.flush()
writer.close()

# final loss with 4 decimals    
final_train_loss = '{:.4f}'.format(epoch_train_loss_full)
final_test_loss = '{:.4f}'.format(epoch_test_loss_full)

print('___Training done!___')


#-------------------------------------------------------------------------------

# save model to folder
torch.save(model, model_path)

print(f'\nModel {model_name} saved in:')
print('\t', model_dir)
print(f'final train loss: {final_train_loss}')
print(f'final test loss: {final_test_loss}')


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

# then training details, from config file used
config_model.add_section('training')
for (each_key, each_val) in config.items('training'):
      config_model.set('training', each_key, each_val)

# where to write it
config_path = model_dir.joinpath(model_name +'.ini') # same name as model
# write
with open(config_path, 'w') as configfile:
    config_model.write(configfile)