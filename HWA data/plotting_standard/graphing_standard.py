## Library Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pltstd_funcs import *

## Define text sizes for **SAVED** pictures (texpsize -- text export size)
texpsize= [26,28,30]

## Input Arrays
x = np.linspace(1,10,10)
y = np.copy(x)
theta = np.linspace(0,90,10)*np.pi/180

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

## Graph -- Standard graph
fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
ax[0,0].plot(x, y, linewidth=2, label="test")
ax[0,0].plot(x+x, y+y, linewidth=2, linestyle="dashed", label="test2")
#ax[0,0].loglog(x, y, marker = "s", color='black', markerfacecolor='none', markeredgewidth=2, markersize=6, label="test")
ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")                        ## String is treatable as latex code
ax[0,0].set_ylabel(r"Deflection in $y^{\prime}$ direction $u\,\,[m]$")
#ax[0,0].set_xlim(0,x[-1])
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
#fig.savefig("yeet.png", bbox_inches='tight')                                    ## Insert save destination
## If you want to see the figure, else disable last two lines.
#fig.tight_layout()
#plt.show()

## Graph -- twin axes
fig2, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
ax[0,0].plot(x, y, linewidth=2, color='r', label=r'$c/R$')
ax[0,0].plot([], [], linewidth=2, color='b', label=r'$\beta$')
ax[0,0].axvspan(0, 2, facecolor='y', alpha=0.2)
ax[0,0].set_xlabel(r"Radial position along blade $r/R\,\,[-]$")
ax[0,0].set_ylabel(r"Chord length $c/R\,\,[-]$")          ## String is treatable as latex code
#ax[0,0].set_xlim(0,1)
#ax[0,0].set_ylim(0.074,0.154)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax2=ax[0,0].twinx()
ax2.plot(x, y*y, linewidth=2, color='b', label=r'$\beta$')
ax2.set_ylabel(r"Pitch angle $\beta\,\,[^{\circ}]$")          ## String is treatable as latex code
#ax2.set_ylim(11.5,37.5)
ax2.minorticks_on()
ax2.tick_params(which='major', length=10, width=2, direction='inout')
ax2.tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
align_yaxes(ax[0,0],ax2)
#fig.savefig("blade_geo.png", bbox_inches='tight')

## Graph -- polar plot (rough)
fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9),subplot_kw={'projection':'polar'})
ax[0,0].plot(theta, np.sqrt(x*x+y*y), marker='x', markersize=8, linewidth=2)
fig.text(0.46,0.02,r"$u_s/u_{se}\,\,\,\,[-]$", size=SMALL_SIZE)          ## 'x-axis' label
ax[0,0].set_ylabel(r"$u_n/u_{se}\,\,\,\,[-]$")
thetaticks = np.arange(0,90+15,15)
ax[0,0].set_thetalim(0,np.pi/2)
ax[0,0].set_thetagrids(thetaticks)
ax[0,0].set_rlim(0,15)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(direction='out', axis='x', pad=20) ## x corresponds to theta
generate_polar_ticks(ax[0,0], 'major', step=15, fullcircle=False)
generate_polar_ticks(ax[0,0], 'minor', step=15/4, fullcircle=False)
generate_radial_ticks(ax[0,0], 'major', orientation='both', step=2)   # [horizontal, vertical, both]
generate_radial_ticks(ax[0,0], 'minor', orientation='both', step=2/4)
#ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
#fig.savefig("us_un.png", bbox_inches='tight')          ## Insert save destination

## Graph -- 360 polar plot (rough)

plt.rc('xtick', labelsize=3*SMALL_SIZE/4)
plt.rc('ytick', labelsize=3*SMALL_SIZE/4)
fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9),subplot_kw={'projection':'polar'})
ax[0,0].plot(theta, np.sqrt(x*x+y*y), marker='x', markersize=8, linewidth=2)
thetaticks = np.arange(0,360,15)
ax[0,0].set_thetalim(0,2*np.pi)
ax[0,0].set_thetagrids(thetaticks)
ax[0,0].set_rlim(0,15)
ax[0,0].set_rlabel_position(90)
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(direction='out', axis='x', pad=10) ## x corresponds to theta
generate_polar_ticks(ax[0,0], 'major', step=15)
generate_polar_ticks(ax[0,0], 'minor', step=15/4)
generate_radial_ticks(ax[0,0], 'major', orientation='both', step=2)   # [horizontal, vertical, both]
generate_radial_ticks(ax[0,0], 'minor', orientation='both', step=2/4)
#ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
#fig.savefig("us_un.png", bbox_inches='tight')          ## Insert save destination
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)

plt.show()
