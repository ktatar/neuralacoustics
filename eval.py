"""Model evaluation script

This script does the following:
1. Load consecutive data points from a single data entry
2. Load specified model and checkpoint for evaluation
3. Visualize full domain output of model prediction, ground truth label, and their differences.
4. If iterations > 2 and valid microphone position is provided, 
   visualize predicted and ground truth waveform picked up by mic,
   and write the recorded audio as pred.wav
"""

import torch

from pathlib import Path
from timeit import default_timer
from sys import platform

import math
from scipy.io.wavfile import write
import numpy as np

# to load model structure
from networks.FNO2d.FNO2d import FNO2d
# to load dataset
from neuralacoustics.DatasetManager import DatasetManager
# to plot data entries (specific series of domains)
from neuralacoustics.data_plotter import plot2Domains, plot3Domains, plotWaveform
# for PyTorch DataLoader determinism
from neuralacoustics.utils import openConfig, getConfigParser, seed_worker, getProjectRoot
# for operation count
# from thop import profile, clever_format (not used)
# from ptflops import get_model_complexity_info (not used)
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str

# Retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)

# Get configuration parameters for evaluation
config, _ = getConfigParser(prj_root, __file__)

# Dataset
dataset_name = config['evaluation'].get('dataset_name')
dataset_dir = config['evaluation'].get('dataset_dir')
dataset_dir = dataset_dir.replace('PRJ_ROOT', prj_root)

# Evaluation setting
entry = config['evaluation'].getint('entry')
offset = config['evaluation'].getint('offset')
iterations = config['evaluation'].getint('iterations')
pause_sec = config['evaluation'].getfloat('pause_sec')

mic_x = config['evaluation'].getint('mic_x')
mic_y = config['evaluation'].getint('mic_y')

dev = config['evaluation'].get('dev')
seed = config['evaluation'].getint('seed')

opcount = config['evaluation'].getint('opcount')

# Model
model_root = config['evaluation'].get('model_dir')
model_root = model_root.replace('PRJ_ROOT', prj_root)

model_name = config['evaluation'].get('model_name')
checkpoint_name = config['evaluation'].get('checkpoint_name')

if checkpoint_name != "":
    model_path = Path(model_root).joinpath(model_name).joinpath('checkpoints').joinpath(checkpoint_name)
else:
    model_path = Path(model_root).joinpath(model_name).joinpath(model_name)


# Load model structure parameters
model_ini_path = Path(model_root).joinpath(model_name).joinpath(model_name+'.ini')
model_config = openConfig(model_ini_path, __file__)

T_in = model_config['training'].getint('T_in')
T_out = model_config['training'].getint('T_out')

# Load normalization info
normalize = model_config['training'].getint('normalize_data')

# Load network object
network_name = model_config['training'].get('network_name')
network_dir_ = model_config['training'].get('network_dir')
network_dir = Path(network_dir_.replace('PRJ_ROOT', prj_root)) / network_name
network_path = network_dir / (network_name + '.py')

network_config_path = model_ini_path

# Read network inferrence type
network_config = openConfig(network_config_path, __file__)
inference_type = network_config['network_details'].get('inference_type')

# Load network
network_path_folders = network_path.parts
network_path_struct = '.'.join(network_path_folders)[:-3]
network_mod = __import__(network_path_struct, fromlist=['*'])
network = getattr(network_mod, network_name)

# Determinism (https://pytorch.org/docs/stable/notes/randomness.html)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

# Determinism for DataLoader
# for permutation in loadDataset() and  seed_worker() in utils.py
torch.manual_seed(seed)
g = torch.Generator()
g.manual_seed(seed)

#---------------------------------------------------------------------
# Load model
print(f"Load model: {model_name}")

# Use the last checkpoint if the provided checkpoint is not valid---we assume that path may be invalid only if checkpoint is passed
if not model_path.is_file():
    checkpoint_path = Path(model_root).joinpath(model_name).joinpath('checkpoints')
    checkpoints = [x.name for x in list(checkpoint_path.glob('*'))]
    checkpoints.sort()
    model_path = checkpoint_path.joinpath(checkpoints[-1])
elif checkpoint_name != "":
    print(f"\tcheckpoint: {checkpoint_name}")

a_normalizer = None
y_normalizer = None
if normalize:
    a_normalizer = torch.load(model_path)['a_normalizer']
    y_normalizer = torch.load(model_path)['y_normalizer']

if dev == 'gpu' or 'cuda' in dev:
    assert(torch.cuda.is_available())
    dev = torch.device('cuda')
    model = network(network_config_path, T_in).cuda()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
else:
    dev = torch.device('cpu')
    model = network(network_config_path, T_in)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])


#---------------------------------------------------------------------
# Load entry from dataset [test set]

data_t_out = None
if inference_type == 'multiple_step':
    data_t_out = T_out
else:
    data_t_out = 1

dataset_manager = DatasetManager(dataset_name, dataset_dir, False)
u = dataset_manager.loadDataEntry(n=iterations, 
                                  win=T_in+data_t_out, 
                                  entry=entry, 
                                  stride=data_t_out, 
                                  offset=offset)

if opcount:
    u_opcount = dataset_manager.loadDataEntry(n=1, win=T_in+data_t_out, entry=0)
    a_opcount = u_opcount[:, :, :, :T_in]

# Get domain size
u_shape = list(u.shape)
S = u_shape[1]
iterations = u_shape[0] # reload iterations

# Assume that all datasets have simulations spanning square domains
assert(S == u_shape[2])

# Prepare test set
n_test = iterations
test_a = u[0:1, :, :, :T_in] # auto-regressive, hence only need 1 initial input is needed
test_u = u[-n_test:, :, :, T_in:T_in+data_t_out]

#print(train_u.shape, test_u.shape)
assert(S == test_u.shape[-2])
assert(data_t_out == test_u.shape[-1])

# Set plot_waveform flag to true if mic position is valid and iterations >= 2
plot_waveform = mic_x >= 0 and mic_y >= 0 and u_shape[0] >= 2
if plot_waveform:
    # Check validity of mic_x and mic_y
    if mic_x >= S or mic_y >= S:
        raise AssertionError("mic_x/mic_y out of bound")

    pred_waveform = torch.zeros(iterations * data_t_out + T_in)
    label_waveform = torch.zeros(iterations * data_t_out + T_in)
    pred_waveform[:T_in] = test_a[0, mic_x, mic_y, :T_in]
    label_waveform[:T_in] = test_a[0, mic_x, mic_y, :T_in]

# Normalize input data 
if normalize:
    print("Normalizing input data...")
    test_a = a_normalizer.encode(test_a)

if inference_type == 'multiple_step':
    test_a = test_a.reshape(1, S, S, 1, T_in).repeat([1, 1, 1, data_t_out, 1])  
else:
    test_a = test_a.reshape(1, S, S, T_in)

if platform == 'darwin':
    num_workers = 0 
else:
    num_workers = 1 # for now single-process data loading, called explicitly to assure determinism in future multi-process calls

print(f'Test input shape: {test_a.shape}, output shape: {test_u.shape}')

print('Evaluation parameters:')
print(f'\tdataset name: {dataset_name}')
print(f'\trequested entry: {entry}')
print(f'\trequested iterations: {iterations}')
print(f'\ttotal timesteps: {iterations * data_t_out + T_in}')
print(f'\tmic_x: {mic_x}')
print(f'\tmic_y: {mic_y}')
print(f'\trandom seed: {seed}\n')

#---------------------------------------------------------------------
# Start evaluation

if normalize:
    if dev == torch.device('cuda'):
        y_normalizer.cuda()
    else:
        y_normalizer.cpu()

model.eval()
with torch.no_grad():
    for i in range(iterations):
        if i == 0:
            features = test_a.to(dev)
            print(features.shape)
        
        label = test_u[i:i+1, ...].to(dev)

        t1 = default_timer()
        prediction = model(features)
        t2 = default_timer()
        print(f'Iteration {i + 1} of {iterations}, inference computation time: {t2 - t1}s')
        
        if inference_type == 'multiple_step':
            prediction = prediction.view(1, S, S, data_t_out)
            
            if plot_waveform:
                pred_waveform[T_in+i*data_t_out:T_in+(i+1)*data_t_out] = prediction[0, mic_x, mic_y, :]
                label_waveform[T_in+i*data_t_out:T_in+(i+1)*data_t_out]= label[0, mic_x, mic_y, :]

            if normalize:
                prediction = y_normalizer.decode(prediction)

            # Only plot domain with small iteration number
            if iterations <= 5:
                for k in range(data_t_out):
                    domains = torch.stack([prediction[0, :, :, k], 
                                           label[0, :, :, k], 
                                           prediction[0, :, :, k] - label[0, :, :, k]])
                    titles = ['Prediction', 'Ground Truth', 'Difference']
                    plot3Domains(domains, pause=pause_sec,
                                 figNum=1, titles=titles, mic_x=mic_x, mic_y=mic_y)
            
            # Auto-regressive    
            features = prediction[:, :, :, -T_in:].reshape(1, S, S, 1, T_in).repeat([1, 1, 1, data_t_out, 1])  
            
        else:
            if plot_waveform:
                pred_waveform[T_in + i] = prediction[0, mic_x, mic_y, 0]
                label_waveform[T_in + i] = label[0, mic_x, mic_y, 0]

            # pred_squared = torch.square(prediction)
            # pred_energy = torch.sum(pred_squared)
            
            # label_squared = torch.square(label)
            # label_energy = torch.sum(label_squared)
            
            # print(f"Prediction energy: {pred_energy}\tlabel energy: {label_energy}")
            
            # Only plot domain with small iteration number
            if iterations <= 200:
                domains = torch.stack([prediction, label, prediction - label]) # prediction shape: [1, 64, 64, 1]
                titles = ['Prediction', 'Ground Truth', 'Difference']
                plot3Domains(domains[:, 0, ..., 0], pause=pause_sec,
                            figNum=1, titles=titles, mic_x=mic_x, mic_y=mic_y)

            # Auto-regressive
            features = torch.cat((features[..., 1:], prediction), -1)

    # Plot waveform
    if plot_waveform:
        plotWaveform(data=torch.stack([pred_waveform, label_waveform]), titles=[
                    'Prediction', 'Ground Truth'])
        sample_arr_pred = pred_waveform.cpu().detach().numpy()
        sample_arr_label = label_waveform.cpu().detach().numpy()
        # TODO retrieve sample rate from dataset
        write('pred.wav', 44100, sample_arr_pred)
        write('label.wav', 44100, sample_arr_label)
    
    # Count operation number using the input
    if opcount:
        a_opcount = a_opcount.to(dev)
        
        # fvcore
        flops_alt = FlopCountAnalysis(model, a_opcount)
        print(f'Operation count:')
        print(flop_count_table(flops_alt))
        # print(flops_alt.by_module_and_operator())
        
        # pytorch-OpCounter
        # flops, params = profile(model=model, inputs=(a_opcount,))
        # print(f'Operation count:')
        # print(f'\tflops: {flops}, params: {params}')

        # flops-counter
        # macs, params = get_model_complexity_info(model, (256, 256, 10), as_strings=True, print_per_layer_stat=True, verbose=True)
        # print(f"Complexity: {macs}")
        # print(f"Parameters: {params}")


# with torch.no_grad():
#     for i, (features, label) in enumerate(test_loader):
#         features = features.to(dev)
#         label = label.to(dev)

#         t1 = default_timer()
#         prediction = model(features)
#         t2 = default_timer()
#         print(f'Timestep {i} of {timesteps}, inference computation time: {t2 - t1}s')

#         prediction = prediction.view(1, S, S, 40)

#         if normalize:
#             prediction = y_normalizer.decode(prediction)

#         for i in range(40):
#             domains = torch.stack([prediction[0, :, :, i], label[0, :, :, i], prediction[0, :, :, i] - label[0, :, :, i]])
#             titles = ['Prediction', 'Ground Truth', 'Difference']
#             plot3Domains(domains, pause=pause_sec,
#                         figNum=1, titles=titles, mic_x=mic_x, mic_y=mic_y)
