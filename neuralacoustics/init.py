import torch
import torch.nn as nn
from torchaudio import transforms
from torch import Tensor
import configparser, argparse # to read config from ini file
from pathlib import Path # to properly handle paths and folders on every os

import numpy as np
import random as rd
import scipy.io
import h5py
import operator
