import torch

from pathlib import Path
from timeit import default_timer

from neuralacoustics.model import FNO2d
from neuralacoustics.dataset_loader import loadDataset # to load dataset
from neuralacoustics.data_plotter import plotDomain # to plot data entries (specific series of domains)
from neuralacoustics.data_plotter import plot2Domains, plot3Domains # to plot data entries (specific series of domains)
from neuralacoustics.utils import seed_worker # for PyTorch DataLoader determinism
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser


# Retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)

# Get configuration parameters for evaluation
config = getConfigParser(prj_root, __file__)

# Dataset
dataset_name = config['evaluation'].get('dataset_name')
dataset_dir = config['evaluation'].get('dataset_dir')
dataset_dir = dataset_dir.replace('PRJ_ROOT', prj_root)

T_in = config['evaluation'].getint('T_in')
T_out = config['evaluation'].getint('T_out')

# Evaluation setting
entry = config['evaluation'].getint('entry')
timesteps = config['evaluation'].getint('timesteps')
pause_sec = config['evaluation'].getint('pause_sec')

mic_x = config['evaluation'].getfloat('mic_x')
mic_y = config['evaluation'].getfloat('mic_y')

dev = config['evaluation'].get('dev')
seed = config['evaluation'].getint('seed')

# Model
model_root = config['evaluation'].get('model_dir')
model_root = model_root.replace('PRJ_ROOT', prj_root)

model_name = config['evaluation'].get('model_name')
checkpoint_name = config['evaluation'].get('checkpoint_name')

model_path = Path(model_root).joinpath(model_name).joinpath('checkpoints').joinpath(checkpoint_name)

# Load model structure parameters
model_ini_path = Path(model_root).joinpath(model_name).joinpath(model_name+'.ini')
if model_ini_path.is_file():
    model_config = getConfigParser(model_ini_path, __file__)

    network_mode = model_config['training'].getint('network_modes')
    network_width = model_config['training'].getint('network_width')
else:
    print("Cannot find model .ini file, using default structure parameters")
    network_mode = 12
    network_width = 20


# Determinism (https://pytorch.org/docs/stable/notes/randomness.html)
torch.use_deterministic_algorithms(True) 
torch.backends.cudnn.deterministic = True 

# Determinism for DataLoader 
torch.manual_seed(seed) # for permutation in loadDataset() and  seed_worker() in utils.py
g = torch.Generator()
g.manual_seed(seed)

# VIC if dataset has nsteps >= 20, here we extract from a single entry [simulation] 10 consecutive datapoints
u = loadDataset(dataset_name=dataset_name,
                dataset_root=dataset_dir,
                n=timesteps,
                win=T_in+T_out,
                stride=1,
                win_lim=0,
                start_ch=0,
                entry=entry,
                permute=False)

# Get domain size
u_shape = list(u.shape)
S = u_shape[1] 
# Assume that all datasets have simulations spanning square domains
assert(S == u_shape[2])

# Prepare test set
n_test = timesteps
test_a = u[-n_test:, :, :, :T_in]
test_u = u[-n_test:, :, :, T_in:T_in+T_out]

#print(train_u.shape, test_u.shape)
assert(S == test_u.shape[-2])
assert(T_out == test_u.shape[-1])

test_a = test_a.reshape(n_test, S, S, T_in)

num_workers = 1 # for now single-process data loading, called explicitly to assure determinism in future multi-process calls

# Dataloader
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u),
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          worker_init_fn=seed_worker,
                                          generator=g)
print(f'Test input shape: {test_a.shape}, output shape: {test_u.shape}')    

# Load model

if dev == 'gpu' or 'cuda' in dev:
    assert(torch.cuda.is_available())
    dev = torch.device('cuda')
    model = FNO2d(network_mode, network_mode, network_width, T_in).cuda()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
else:
    dev = torch.device('cpu')
    model = FNO2d(network_mode, network_mode, network_width, T_in)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

model.eval()
print(f"Load model from path: {model_path}")

# Start evaluation
for features, label in test_loader:
    features = features.to(dev)
    label = label.to(dev)

    # t1 = default_timer()
    prediction = model(features)
    # t2 = default_timer()
    # print(f'\nInference step computation time: {t2 - t1}s')

    # subplots
    # domains = torch.stack([prediction, label])
    # titles = ['Prediction','Ground Truth']
    # plot2Domains(domains[:,0,...,0], pause=pause_sec, figNum=1, titles=titles)

    domains = torch.stack([prediction, label, prediction - label])
    titles = ['Prediction','Ground Truth', 'Diff']
    plot3Domains(domains[:,0,...,0], pause=pause_sec, figNum=1, titles=titles)

    # two different windows
    # plotDomain(prediction[0,...,0], pause=pause, figNum=2)
    # plotDomain(label[0,...,0], pause=pause, figNum=1)

    # TODO: record waveform by microphone

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