# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 20:29:13 2021

@author: elraw
"""

import numpy as np 
import os 
from matplotlib import pyplot as plt
import statsmodels.api as sm


def get_velocity(coeffcients,volt):
    
    a = coeffcients[0]
    b = coeffcients[1]
    c = coeffcients[2]
    d = coeffcients[3]
    e = coeffcients[4]
    return( a*volt**4+ b*volt**3+ c*volt**2 + d*volt+ e)


def rms_vel (velcoity):
    alphas, positions, samples = velcoity.shape[0], velcoity.shape[1], velcoity.shape[2]
    RMS = np.zeros((alphas,positions))
    for i in range(alphas):
        for j in range(positions):
            measurement  =velocity[i][j]
            mean= np.mean(measurement)
            RMS[i][j] = np.sqrt(np.sum((measurement - mean)**2)/(samples-1))
    
    return(RMS)



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

# plotting the calibration curve     
cal_coeffcient = np.polyfit(volt_cal,vel_cal,deg = 4)
x_volt = np.linspace(np.min(volt_cal),np.max(volt_cal),100)
velo_cal_test = get_velocity(cal_coeffcient,x_volt)

plt.figure(10)
plt.plot(velo_cal_test, x_volt, label = r"$4^{\mathrm{th}}$-order polynomial")
plt.plot(vel_cal, volt_cal, 'ko', label = "Calibration data ")
plt.xlabel(r"${U}$ [m/s]" )
plt.ylabel(r"$E$ [V]")
plt.legend()
plt.savefig("./Figures/calibration.png")
plt.show()


#correlation and sampling rate 
correlation_data = np.genfromtxt('CorrelationTest', skip_header = 23)
time_corr = correlation_data[:,0]
volt_corr = correlation_data[:,1]


#compute the auto corrolation and plot it for different values of shift
#compute the shift needed to make the auto-corrolation is zero
shift = 36
correlated = True
while correlated == True:
    auto_corr=  sm.tsa.acf(volt_corr, nlags = shift, fft = False)[-1]
    print(shift)
    if auto_corr<0:
        correlated = False
    else:
      shift += 1

#compute the required sampling rate
f = 1e3
time_scale = shift/f
sample_rate  = 1/(2*time_scale)


#plot autocorrelation for different shift
shift_plot = shift+300
auto_corr_f = sm.tsa.acf(volt_corr, nlags = shift_plot, fft = False)
N_corr = len(auto_corr_f)
plt.plot(time_corr[0:N_corr],auto_corr_f)
plt.xlabel(r"lag $\tau$ [s]")
plt.ylabel(r"$\rho$($\tau$ )[-]")
plt.grid()
plt.savefig("./Figures/autocorrelation"+".png")
plt.show()





#plot velocities and RMS
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
    plt.xlabel(r"${U}$ [m/s]")
    plt.ylabel("y [mm]")
    plt.grid()
    plt.legend() 
    plt.savefig("./Figures/velocity_"+str(alpha)+".png")
plt.show()
    


        
velcoity_rms= rms_vel(velocity)
for i, alpha in enumerate([0,5,15]):
    plt.figure(i)
    plt.plot( velcoity_rms[i], positions, 'ko-', label= r"$\alpha$ = " +str(alpha))
    plt.xlabel(r"${U}_{\mathrm{RMS}}$ [-]")
    plt.ylabel("y [mm]")
    plt.legend() 
    plt.grid()
    plt.savefig("./Figures/RMS_velocity_"+str(alpha)+".png")

plt.show()
    








