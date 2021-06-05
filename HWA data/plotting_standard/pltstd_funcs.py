"""
Author: Andrea Bettini

This file contains functions used to help in plotting.
"""
## Library Imports
import numpy as np
import matplotlib
import math as m

## Functions
def align_yaxes(ax1,ax2):
    """
    Align the vertical axes of a graph when .twinx() is used.
    Input:
    ax1 -- this is the axis that ax2 will be align itself with.
    ax2 -- the axis that will align with ax1.
    
    Solution taken from
    https://stackoverflow.com/questions/45037386/trouble-aligning-ticks-for-matplotlib-twinx-axes
    """
    # Determine which plot has finer grid. Set pointers accordingly
    l = ax1.get_ylim()
    l2 = ax2.get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(ax1.get_yticks())
    ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    
    return

def generate_polar_ticks(ax, ticktype, step=5, fullcircle=True):
    """
    Generate minor ticks along the circular part of the outer edge of the polar grid
    Input:
    ax -- the plot that the function will generate the minor ticks for.
    ticktype -- creates major or minor ticks ['major', 'minor']
    fullcircle -- is the plot generated a full circle or not? [True, False]
    step -- the step of each minor tick -- basically how many degrees apart each minor tick is placed at. [deg]

    Solution inspired from
    https://stackoverflow.com/questions/44657003/how-to-create-minor-ticks-for-polar-plot-matplotlib
    """

    # Check whether full circle or not, then find the bounding case for the ticks.
    if fullcircle:
        tick_lb = 0     # lower bound (lb)
        tick_ub = 360   # upper bound (ub)
    else:
        tick_lb = ax.get_xticks()[0] * 180/np.pi 
        tick_ub = ax.get_xticks()[-1] * 180/np.pi

    # Draw ticks
    if ticktype == 'minor':
        tick = [ax.get_rmax()*0.99,ax.get_rmax()]
        for t in np.deg2rad(np.arange(tick_lb,tick_ub+step,step)):
            ax.plot([t,t], tick, linewidth=2, color="k")

    elif ticktype == 'major':
        tick = [ax.get_rmax()*0.98,ax.get_rmax()]
        for t in np.deg2rad(np.arange(tick_lb,tick_ub+step,step)):
            ax.plot([t,t], tick, linewidth=2, color="k")

    return

# WORK IN PROGRESS
def generate_radial_ticks(ax, ticktype, orientation='both', step=5):
    """
    Generate minor ticks along the circular part of the outer edge of the polar grid
    Input:
    ax -- the plot that the function will generate the minor ticks for.
    ticktype -- creates major or minor ticks ['major', 'minor']
    orientation -- creates ticks along vertical or horizontal direction or both ['vertical','horizontal','both']
    step -- the step of each minor tick -- basically how many degrees apart each minor tick is placed at. [deg]

    Solution inspired from
    https://stackoverflow.com/questions/44657003/how-to-create-minor-ticks-for-polar-plot-matplotlib
    """

    # Find the bounding case for the ticks.
    tick_lb = ax.get_rmin() # lower bound (lb)
    tick_ub = ax.get_rmax() # upper bound (ub)

    # Set up how many axes to draw ticks for
    if orientation == 'both':
        phi0 = [0, np.pi/2, np.pi, 3*np.pi/2]
    elif orientation == 'horizontal':
        phi0 = [0, np.pi]
    elif orientation == 'vertical':
        phi0 = [np.pi/2, 3*np.pi/2]

    # Draw ticks
    if ticktype == 'minor':
        l = 0.01 * 0.5*ax.get_rmax()
        for r in np.arange(tick_lb,tick_ub+step,step)[1:]:
            phi = m.atan(l/r)
            R = r/m.cos(phi)
            for p0 in phi0:
                ax.plot([p0+phi,p0-phi], [R,R], linewidth=2, color="k")
    
    elif ticktype == 'major':
        l = 0.02 * 0.5*ax.get_rmax()
        for r in np.arange(tick_lb,tick_ub+step,step)[1:]:
            phi = m.atan(l/r)
            R = r/m.cos(phi)
            for p0 in phi0:
                ax.plot([p0+phi,p0-phi], [R,R], linewidth=2, color="k")

    return
    
