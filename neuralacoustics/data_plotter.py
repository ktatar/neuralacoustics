import torch
import matplotlib.pyplot as plt # to plot
import matplotlib.colors as mcolors # to potentially use different color colormaps


def plotDomain(data, color_halfrange=1, maxAmp=20.0, log_min=10.0, pause=0.001, figNum=0):

  img = prepareImgPlot(data, maxAmp, log_min)

  if(figNum > 0):
    plt.figure(figNum)

  plt.imshow(img, vmin=-color_halfrange, vmax=color_halfrange)
  #plt.imshow(img, vmin=-color_halfrange, vmax=color_halfrange, cmap=hdh_cmap) # custom colormap
  # all this non-sense is needed to have a proper non-blocking plot
  # plt.ion()
  # plt.show()
  plt.pause(pause)


def plot2Domains(data, color_halfrange=1, maxAmp=20.0, log_min=10.0, pause=0.001, figNum=0, titles=None):
  img1 = prepareImgPlot(data[0], maxAmp, log_min)
  img2 = prepareImgPlot(data[1], maxAmp, log_min)


  if(figNum > 0):
    fig = plt.figure(figNum)
  else:
    fig = plt.figure()


  rows = 1
  columns = 2
  
  # first plot
  fig.add_subplot(rows, columns, 1)
    
  plt.imshow(img1, vmin=-color_halfrange, vmax=color_halfrange)
  #plt.axis('off')
  if titles != None:
    plt.title(titles[0])
    
  # Adds a subplot at the 2nd position
  fig.add_subplot(rows, columns, 2)
    
  # showing image
  plt.imshow(img2, vmin=-color_halfrange, vmax=color_halfrange)
  #plt.axis('off')
  if titles != None:
    plt.title(titles[1])

  plt.pause(pause)


def plot3Domains(data, color_halfrange=1, maxAmp=20.0, log_min=10.0, pause=0, figNum=0, titles=None):
  img1 = prepareImgPlot(data[0], maxAmp, log_min)
  img2 = prepareImgPlot(data[1], maxAmp, log_min)
  img3 = prepareImgPlot(data[2], maxAmp, log_min)

  if(figNum > 0):
    fig = plt.figure(figNum)
  else:
    fig = plt.figure()

  rows = 1
  columns = 3
  
  # First plot
  fig.add_subplot(rows, columns, 1)
  plt.imshow(img1, vmin=-color_halfrange, vmax=color_halfrange)
  #plt.axis('off')
  if titles != None:
    plt.title(titles[0])
    
  # Second plot
  fig.add_subplot(rows, columns, 2)
  plt.imshow(img2, vmin=-color_halfrange, vmax=color_halfrange)
  #plt.axis('off')
  if titles != None:
    plt.title(titles[1])

  # Third plot
  fig.add_subplot(rows, columns, 3)
  plt.imshow(img3, vmin=-color_halfrange, vmax=color_halfrange)
  #plt.axis('off')
  if titles != None:
    plt.title(titles[2])

  # Proceed through timer or user input
  if pause == 0:
    plt.waitforbuttonpress()
  else:
    plt.pause(pause)
  

#-------------------------------------------------------------------------------------------------------


def prepareImgPlot(data, maxAmp, log_min):
  log_data = torch.abs(data / maxAmp)
  log_data = (torch.log(log_data + 1e-8) + log_min)/log_min
  log_data = torch.clamp(log_data, 0, 1)*torch.sign(data)

  img = log_data.cpu().detach().numpy()
  return img
