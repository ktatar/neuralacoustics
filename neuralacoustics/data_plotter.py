import torch
import matplotlib.pyplot as plt # to plot
import matplotlib.colors as mcolors # to potentially use different color colormaps


def plotDomain(data, color_halfrange=1, maxAmp=20.0, log_min=10.0, pause=0.001):
  log_data = torch.abs(data / maxAmp)
  log_data = (torch.log(log_data + 1e-8) + log_min)/log_min
  log_data = torch.clamp(log_data, 0, 1)*torch.sign(data)

  img = data.cpu().detach().numpy()
  plt.imshow(img, vmin=-color_halfrange, vmax=color_halfrange)
  #plt.imshow(img, vmin=-color_halfrange, vmax=color_halfrange, cmap=hdh_cmap) # custom colormap
  # all this non-sense is needed to have a proper non-blocking plot
  plt.ion()
  plt.show()
  plt.pause(pause)