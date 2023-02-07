import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from neuralacoustics.utils import openConfig

import numpy as np
#VIC this is the content of: https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d_time.py
# i made a small modification to the original code, highlighted by the following comment: #VIC-mod

################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()

        """
        MLP layer that does 1D convolution.    
        """

        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    # WYNN-mod: Add stacks_num input argument
    def __init__(self, config_path, t_in):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous t_in timesteps + 2 locations (u(t-t_in, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=t_in+2)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        # Parse config file
        network_config = openConfig(config_path, __file__)
        self.modes1 = network_config['network_parameters'].getint('network_modes')
        self.modes2 = network_config['network_parameters'].getint('network_modes')
        self.width = network_config['network_parameters'].getint('network_width')
        self.stacks_num = network_config['network_parameters'].getint('stacks_num')
        self.padding = network_config['network_parameters'].getint('padding') # pad the domain if input is non-periodic
        self.mlp_on = network_config['network_parameters'].getboolean('mlp_on')
        self.normalization_on = network_config['network_parameters'].getboolean('normalization_on')
        self.mlp_q = network_config['network_parameters'].getboolean('mlp_q')
        
        #VIC-mod t_in is passed as parameter now, so that we can decide the number of input time steps
        #self.fc0 = nn.Linear(12, self.width)
        self.fc0 = nn.Linear(t_in+2, self.width)
        # input channel is 12: the solution of the previous t_in timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        # WYNN-mod: A module list for stacking layers
        self.conv_list = nn.ModuleList([SpectralConv2d_fast(self.width,
                                                            self.width,
                                                            self.modes1,
                                                            self.modes2) 
                                        for i in range(self.stacks_num)])
        self.w_list = nn.ModuleList([nn.Conv2d(self.width, 
                                               self.width,
                                               1)
                                     for i in range(self.stacks_num)])
        # self.bn_list = nn.ModuleList([nn.BatchNorm2d(self.width) for i in range(self.stacks_num)])
        self.mlp_list = None
        if self.mlp_on:
            self.mlp_list = nn.ModuleList([MLP(self.width,
                                               self.width,
                                               self.width)
                                           for i in range(self.stacks_num)])
        self.norm = None
        if self.normalization_on:
            self.norm = nn.InstanceNorm2d(self.width)
        
        if self.mlp_q:
            self.q = MLP(self.width, 1, self.width * 4)
        else:
            self.fc1 = nn.Linear(self.width, 128)
            self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        if self.padding > 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        # WYNN-mod: Use iteration to forward pass x through the stacked layers
        for i in range(len(self.conv_list)):
            if self.normalization_on:
                x1 = self.norm(self.conv_list[i](self.norm(x)))
            else:
                x1 = self.conv_list[i](x)

            if self.mlp_on:
                x1 = self.mlp_list[i](x1)

            x2 = self.w_list[i](x) 
            x = x1 + x2
            if i != len(self.conv_list) - 1:
                x = F.gelu(x)
        
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        if self.mlp_q:
            x = self.q(x)
            x = x.permute(0, 2, 3, 1)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)