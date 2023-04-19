from calendar import c
import torch
import numpy as np
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

  # Proceed through timer or user input
  if pause == 0:
    pause = 0.00001 #VIC 0 stalls plot
  if pause == -1:
    plt.waitforbuttonpress()
    plt.pause(0.0001) #VIC without this, figure is not updated
  else:
    plt.pause(pause)
  plt.draw() # redraw fig, clear axes
  plt.cla() # from https://stackoverflow.com/questions/10272478/why-does-my-pylab-animation-slow-down-with-each-update

  


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

  if pause == 0:
    pause = 0.00001 #VIC 0 stalls plot

  # Proceed through timer or user input
  if pause == -1:
    plt.waitforbuttonpress()
    plt.pause(0.0001) #VIC without this, figure is not updated
  else:
    plt.pause(pause)
  plt.cla()


def plot3Domains(data, color_halfrange=1, maxAmp=20.0, log_min=10.0, pause=0.001, figNum=0, titles=None, mic_x=-1, mic_y=-1):
  """Plot full domain outputs of model prediction, ground truth and their difference."""
  img1 = prepareImgPlot(data[0], maxAmp, log_min)
  img2 = prepareImgPlot(data[1], maxAmp, log_min)
  img3 = prepareImgPlot(data[2], maxAmp, log_min)

  plot_waveform = mic_x >= 0 and mic_y >= 0
  
  # Double check the mic position is in bound
  if plot_waveform:
    if mic_x >= img1.shape[0] or mic_y >= img1.shape[1]:
      raise AssertionError("mic_x/mic_y out of bound")

  if(figNum > 0):
    fig = plt.figure(figNum)
  else:
    fig = plt.figure()

  rows = 1
  columns = 3

  if plot_waveform:
    highlight_points = []
    if mic_x > 0: highlight_points.append([mic_x - 1, mic_y])
    if mic_x < img1.shape[0] - 1: highlight_points.append([mic_x + 1, mic_y])
    if mic_y > 0: highlight_points.append([mic_x, mic_y - 1])
    if mic_y < img1.shape[1] - 1: highlight_points.append([mic_x, mic_y + 1])

    for pt in highlight_points:
      img1[pt[0], pt[1]] = -1
      img2[pt[0], pt[1]] = -1
  
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

  if pause == 0:
    pause = 0.00001 #VIC 0 stalls plot

  # Proceed through timer or user input
  if pause == -1:
    plt.waitforbuttonpress()
    plt.pause(0.0001) #VIC without this, figure is not updated
  else:
    plt.pause(pause)
  plt.cla()


def plotWaveform(data, sr=44100, maxAmp=20.0, log_min=10.0, titles=None):
  """Plot the prediction and label waveforms collected at microphone position."""
  img1 = prepareImgPlot(data[0], maxAmp, log_min)
  img2 = prepareImgPlot(data[1], maxAmp, log_min)

  # Set x-axis as millisecond
  x = np.array(list(range(img1.shape[0])))
  t = x / sr * 1000

  # Close previous plot
  plt.close()

  fig = plt.figure(1)
  rows = 2
  columns = 1

  # First plot
  fig.add_subplot(rows, columns, 1)
  plt.plot(t, img1, color='red')
  plt.xlabel("Time (ms)")
  plt.ylabel("Amplitude")
  if titles != None:
    plt.title(titles[0]) 
  
  # Second plot
  fig.add_subplot(rows, columns, 2)
  plt.plot(t, img2, color='green')
  plt.xlabel("Time (ms)")
  plt.ylabel("Amplitude")
  if titles != None:
    plt.title(titles[1])
  
  plt.tight_layout()
  plt.show()
  plt.cla()
#-------------------------------------------------------------------------------------------------------


def prepareImgPlot(data, maxAmp, log_min):
  log_data = torch.abs(data / maxAmp)
  log_data = (torch.log(log_data + 1e-8) + log_min)/log_min
  log_data = torch.clamp(log_data, 0, 1)*torch.sign(data)

  img = log_data.cpu().detach().numpy()
  return img

