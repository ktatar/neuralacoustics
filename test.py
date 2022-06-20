import torch
from neuralacoustics.model import FNO2d
from neuralacoustics.dataset_loader import loadDataset # to load dataset
from neuralacoustics.data_plotter import plotDomain # to plot data entries (specific series of domains)
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser


# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)


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