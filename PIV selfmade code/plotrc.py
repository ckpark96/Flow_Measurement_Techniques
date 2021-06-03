import matplotlib.pyplot as plt
import matplotlib

## Define text sizes for **SAVED** pictures (texpsize -- text export size)
texpsize= [26,28,30]

## Graphing Parameters
SMALL_SIZE  = texpsize[0]
MEDIUM_SIZE = texpsize[1]
BIGGER_SIZE = texpsize[2]

plt.style.use('grayscale')
plt.rc('font', size=MEDIUM_SIZE, family='serif')    ## controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
plt.rc('text', usetex=False)
matplotlib.rcParams['lines.linewidth']  = 1.5
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor']   = 'white'
matplotlib.rcParams["legend.fancybox"]  = False