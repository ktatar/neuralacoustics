import torch
from timeit import default_timer

from neuralacoustics.model import FNO2d
from neuralacoustics.dataset_loader import loadDataset # to load dataset
from neuralacoustics.data_plotter import plotDomain # to plot data entries (specific series of domains)
from neuralacoustics.data_plotter import plot2Domains # to plot data entries (specific series of domains)
from neuralacoustics.utils import seed_worker # for PyTorch DataLoader determinism
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser


# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)






dataset_name = 'dataset_2'
model_path = 'models_bu/22-06-27_04-23_c2185/22-06-27_04-23_c2185'

pause = 0.5

seed = 0






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
# load dataset

dataset_dir = 'PRJ_ROOT/datasets'
dataset_dir = dataset_dir.replace('PRJ_ROOT', prj_root)

S = 64
batch_size = 1

#VIC if dataset has nsteps >= 20, here we extract from a single entry [simulation] 10 consecutive datapoints
T_in = 10
T_out = 1
n_test = 10
win_stride = 1
win_limit = 20
start_ch = 0
permute = False

u = loadDataset(dataset_name, dataset_dir, n_test, T_in+T_out, win_stride, win_limit, start_ch, permute)

# get domain size
sh = list(u.shape)
S = sh[1] 
# we assume that all datasets have simulations spanning square domains
assert(S == sh[2])

# prepare test set
test_a = u[-n_test:,:,:,:T_in]
test_u = u[-n_test:,:,:,T_in:T_in+T_out]

#print(train_u.shape, test_u.shape)
assert(S == test_u.shape[-2])
assert(T_out == test_u.shape[-1])

test_a = test_a.reshape(n_test,S,S,T_in)

num_workers = 1 # for now single-process data loading, called explicitly to assure determinism in future multi-process calls

# datapoints will be loaded from this
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False, 
num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

print(f'Test input shape: {test_a.shape}, output shape: {test_u.shape}')    




#-------------------------------------------------------------------------------
# load model
dev  = torch.device('cpu')
model = torch.load(model_path, dev)



#-------------------------------------------------------------------------------
# test

for features, label in test_loader:
    loss = 0
    features = features.to(dev)
    label = label.to(dev)

    # t1 = default_timer()
    prediction = model(features)
    # t2 = default_timer()
    # print(f'\nInference step computation time: {t2-t1}s')

    # subplots
    domains = torch.stack([prediction, label])
    titles = ['Prediction','Ground Truth']
    plot2Domains(domains[:,0,...,0], pause=pause, figNum=1, titles=titles)
    # two different windows
    # plotDomain(prediction[0,...,0], pause=pause, figNum=2)
    # plotDomain(label[0,...,0], pause=pause, figNum=1)

    # auto-regressive
    features = torch.cat((features, prediction), -1)




#VIC this script is supposed to do the following:

# load consecutive data points from a single data entry
# _choose:
# __dataset
# __data entry [simulation] -> should load only chunck where chosen data entry, to save ram
# __how many consecutive data points [simulation steps] from chosen data entry -> T_in from model.ini, T_out=1, stride=1, win_lim=0, batch_size=1

# visualize difference between prediction and label:
# _plot 3 windows:
# __label window and prediction window
# __difference window
# _use either command line input to go to next data poit or timer

# print in console time that it takes for model to generate prediction

# if data points > 2, can visualize waveform picked up by mic
# _choose mic position
# _sample at every datapoint value at mic pos on both label and prediction window
# _at end of simulation, plot two waveform windows 

# make sure that it works on both cpu and gpu... no sure how...



#VIC 
# test_features, test_labels = next(iter(test_loader))
# print(test_features.size())
# print(test_labels.size())
# t1 = default_timer()
# prediction = model(test_features)
# t2 = default_timer()
# print(f'\nInference step finished, computation time: {t2-t1}s')
# quit()