import torch
from timeit import default_timer
import configparser
from pathlib import Path

from neuralacoustics.model import FNO2d
from neuralacoustics.dataset_loader import loadDataset # to load dataset
from neuralacoustics.utils import LpLoss
from neuralacoustics.utils import getProjectRoot
from neuralacoustics.utils import getConfigParser
from neuralacoustics.data_plotter import plotDomain # to plot data entries (specific series of domains)


# retrieve PRJ_ROOT
prj_root = getProjectRoot(__file__)


#VIC this script is supposed to do the following:
#_load dataset with batch 1 -> note that T_in and T_out can be different than training!
#_load model with torch.load(MODEL_PATH)
#_iterate over all selected data entries and
#__get predictions, as in the case of training [using output as next input]
#__compute step and full loss
#__count elapsed time per each prediction
#__[optional] plot predictions
#_print average losses [like in training] and average prediction time!

# make sure that it works on both cpu and gpu... no sure how...