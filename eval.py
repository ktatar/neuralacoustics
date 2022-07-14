"""Model evaluation script

This script does the following:
1. Load consecutive data points from a single data entry
2. Load specified model and checkpoint for evaluation
3. Visualize full domain output of model prediction, ground truth label, and their differences.
4. If timesteps > 2 and valid microphone position is provided, visualize predicted and label waveform picked up by mic
"""

import torch

from pathlib import Path
from timeit import default_timer

from neuralacoustics.model import FNO2d
# to load dataset
from neuralacoustics.DatasetManager import DatasetManager
# to plot data entries (specific series of domains)
from neuralacoustics.data_plotter import plot2Domains, plot3Domains, plotWaveform
# for PyTorch DataLoader determinism
from neuralacoustics.utils import openConfig, getConfigParser, seed_worker, getProjectRoot


# Retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)

# Get configuration parameters for evaluation
config = getConfigParser(prj_root, __file__)

# Dataset
dataset_name = config['evaluation'].get('dataset_name')
dataset_dir = config['evaluation'].get('dataset_dir')
dataset_dir = dataset_dir.replace('PRJ_ROOT', prj_root)

# Evaluation setting
entry = config['evaluation'].getint('entry')
timesteps = config['evaluation'].getint('timesteps')
pause_sec = config['evaluation'].getfloat('pause_sec')

mic_x = config['evaluation'].getint('mic_x')
mic_y = config['evaluation'].getint('mic_y')
plot_waveform = mic_x >= 0 and mic_y >= 0 and timesteps >= 2

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
model_config = openConfig(model_ini_path, __file__)

network_mode = model_config['training'].getint('network_modes')
network_width = model_config['training'].getint('network_width')
T_in = model_config['training'].getint('T_in')
T_out = model_config['training'].getint('T_out')


# Determinism (https://pytorch.org/docs/stable/notes/randomness.html)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

# Determinism for DataLoader
# for permutation in loadDataset() and  seed_worker() in utils.py
torch.manual_seed(seed)
g = torch.Generator()
g.manual_seed(seed)



#---------------------------------------------------------------------
# load entry from dataset [test set]

dataset_manager = DatasetManager(dataset_name, dataset_dir)
u = dataset_manager.loadDataEntry(n=timesteps, win=T_in+T_out, entry=entry)

# Get domain size
u_shape = list(u.shape)
S = u_shape[1]
# Assume that all datasets have simulations spanning square domains
assert(S == u_shape[2])

if plot_waveform:
    # Check validity of mic_x and mic_y
    if mic_x >= S or mic_y >= S:
        raise AssertionError("mic_x/mic_y out of bound")

    pred_waveform = torch.zeros(timesteps)
    label_waveform = torch.zeros(timesteps)

# Prepare test set
n_test = timesteps
test_a = u[-n_test:, :, :, :T_in]
test_u = u[-n_test:, :, :, T_in:T_in+T_out]

#print(train_u.shape, test_u.shape)
assert(S == test_u.shape[-2])
assert(T_out == test_u.shape[-1])

test_a = test_a.reshape(n_test, S, S, T_in)

num_workers = 1  # for now single-process data loading, called explicitly to assure determinism in future multi-process calls

# Dataloader
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u),
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          worker_init_fn=seed_worker,
                                          generator=g)
print(f'Test input shape: {test_a.shape}, output shape: {test_u.shape}')




#---------------------------------------------------------------------
# Load model

# Use the last checkpoint if the provided checkpoint is not valid
if not model_path.is_file():
    checkpoint_path = Path(model_root).joinpath(model_name).joinpath('checkpoints')
    checkpoints = [x.name for x in list(checkpoint_path.glob('*'))]
    checkpoints.sort()
    model_path = checkpoint_path.joinpath(checkpoints[-1])

if dev == 'gpu' or 'cuda' in dev:
    assert(torch.cuda.is_available())
    dev = torch.device('cuda')
    model = FNO2d(network_mode, network_mode, network_width, T_in).cuda()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
else:
    dev = torch.device('cpu')
    model = FNO2d(network_mode, network_mode, network_width, T_in)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])




#---------------------------------------------------------------------
# model evaluation

model.eval()
print(f"Load model from path: {model_path}")

# Start evaluation
with torch.no_grad():
    for i, (features, label) in enumerate(test_loader):
        features = features.to(dev)
        label = label.to(dev)

        t1 = default_timer()
        prediction = model(features)
        t2 = default_timer()
        print(f'Inference step computation time: {t2 - t1}s')

        if plot_waveform:
            pred_waveform[i] = prediction[0, mic_x, mic_y, 0]
            label_waveform[i] = label[0, mic_x, mic_y, 0]

        domains = torch.stack([prediction, label, prediction - label])
        titles = ['Prediction', 'Ground Truth', 'Diff']
        plot3Domains(domains[:, 0, ..., 0], pause=pause_sec,
                    figNum=1, titles=titles)

        # auto-regressive
        features = torch.cat((features, prediction), -1)

    # Plot waveform
    if plot_waveform:
        plotWaveform(data=torch.stack([pred_waveform, label_waveform]), titles=[
                    'Prediction', 'Ground Truth'])