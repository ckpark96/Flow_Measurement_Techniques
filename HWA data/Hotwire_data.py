# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 20:29:13 2021

@author: elraw
"""

import numpy as np 
import os 
from matplotlib import pyplot as plt

#import statsmodels.api as sm

#import statsmodels.tsa.stattools as sm
#import statsmodels.api as sm 

work_path = os.getcwd()

calibration_path = work_path + "/Calibration"

volt_cal = np.zeros(11)
vel_cal  = np.arange(0,21,2)

for i in range(0,21,2):
    if i < 10: 
        data = np.genfromtxt(calibration_path + "/Calibration_00"+str(i), skip_header = 23)
    else:
        data = np.genfromtxt(calibration_path + "/Calibration_0"+str(i), skip_header  = 23)

    volt_cal[int(i/2)] = np.average(data[:,1])
    

cal_coeffcient = np.polyfit(volt_cal,vel_cal,deg = 4)


def get_velocity(coeffcients,volt):
    
    a = coeffcients[0]
    b = coeffcients[1]
    c = coeffcients[2]
    d = coeffcients[3]
    e = coeffcients[4]
    return( a*volt**4+ b*volt**3+ c*volt**2 + d*volt+ e)

x_volt = np.linspace(np.min(volt_cal),np.max(volt_cal),100)
velo_cal_test = get_velocity(cal_coeffcient,x_volt)
plt.plot(velo_cal_test, x_volt, r"$4^{\mathrm{th}}$-order polynomial")
plt.plot(vel_cal, volt_cal, 'o', label = "Calibration data ")
plt.xlabel( )
plt.ylabel(r"$E [V] ")
plt.show()


correlation_data = np.genfromtxt('CorrelationTest', skip_header = 23)
time_corr = correlation_data[:,0]
volt_corr = correlation_data[:,1]

auto_corr = np.correlate(volt_corr, volt_corr, mode = 'full')/len(volt_corr)
auto_corr = auto_corr[ int (auto_corr.size/2): ]
auto_corr_coeff= np.corrcoef(volt_corr, volt_corr)





plt.figure()
#plt.plot(time_corr, volt_corr)
plt.plot(time_corr, auto_corr)
plt.show()

data_sample = np.genfromtxt(os.path.join(work_path,"0 aoa")+"/Measurement_+00_+00", skip_header = 23)[:,0]
N_data = len(data_sample)

N = 21   # number of files so 21 positions along the airfoil 
voltage = np.zeros((3,N,N_data))   #3 AOA and N files for each angle. 
voltage_mean = np.zeros((3,N))



for i , alpha  in enumerate([0,5,15]):
    
    if alpha<10:
        angle = str(0)
    else: 
        angle = str('')
        
    for j, pos in enumerate(range(-40,44,4)):
        if pos>= 0 and pos<10: 
            voltage_ij =  np.genfromtxt(os.path.join(work_path,str(alpha)+" aoa")+"/Measurement_+0"+str(pos)+"_+"+ angle +str(alpha), skip_header = 23)[:,1]
        elif  pos>= 0 and pos>10: 
            voltage_ij =  np.genfromtxt(os.path.join(work_path,str(alpha)+" aoa")+"/Measurement_+"+str(pos)+"_+"+ angle + str(alpha), skip_header = 23)[:,1]
        elif pos<0 and pos >-10:  
            voltage_ij =  np.genfromtxt(os.path.join(work_path,str(alpha)+" aoa")+"/Measurement_-0"+str(np.abs(pos))+"_+"+ angle +str(alpha), skip_header = 23)[:,1]
        elif pos<0 and pos<-10:
            voltage_ij =  np.genfromtxt(os.path.join(work_path,str(alpha)+" aoa")+"/Measurement_"+str((pos))+"_+"+ angle +str(alpha), skip_header = 23)[:,1]

        voltage[i][j][:] = voltage_ij 
        voltage_mean[i][j] = np.mean(voltage_ij)


velocity =  get_velocity(cal_coeffcient, voltage)
velocity_mean = get_velocity(cal_coeffcient, voltage_mean)
positions = np.linspace(-40,40,N)


for i, alpha in enumerate([0,5,15]):
    plt.figure(i)
    plt.plot( velocity_mean[i],positions, 'ko-', label= r"$\alpha$ = " +str(alpha))
    #plt.plot( velcoity_rms[i], positions, 'o-')
    plt.xlabel(r"${U}$ [-]")
    plt.ylabel("y [mm]")
    plt.grid()
    plt.legend() 
plt.show()
    



def rms_vel (velcoity):
    alphas, positions, samples = velcoity.shape[0], velcoity.shape[1], velcoity.shape[2]
    RMS = np.zeros((alphas,positions))
    for i in range(alphas):
        for j in range(positions):
            measurement  =velocity[i][j]
            mean= np.mean(measurement)
            RMS[i][j] = np.sqrt(np.sum((measurement - mean)**2)/(samples-1))
    
    return(RMS)
        
velcoity_rms= rms_vel(velocity)


for i, alpha in enumerate([0,5,15]):
    plt.figure(i)
    #plt.plot( positions, velocity_mean[i], 'ko-')
    plt.plot( velcoity_rms[i], positions, 'o-')
    plt.xlabel(r"${U}_{\mathrm{RMS}}$ [-]")
    plt.ylabel("y [mm]")
    plt.legend() 
plt.show()
    








