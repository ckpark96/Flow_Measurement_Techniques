from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from numpy.lib.function_base import average
from scipy import signal
from scipy import stats
import scipy
import scipy.io
import os
import multiprocessing
import pickle

"""
TODO: Loop Parallelization (May not even be feasible)
TODO: Proper scaling of values to velocity
TODO: Subpixel interpolation (Done)
"""

def fit_gaussian(image,ic,jc,dx):
    dist = np.zeros_like(image)
    for i in range(dist.shape[1]):
        dist[:,i] = np.arange(0,dist.shape[1],1)

    
    imax = np.sum(dist*image)/np.sum(image)
    jmax = np.sum(dist.T*image)/np.sum(image)

    return imax+(ic-dx),jmax+(jc-dx)

def process_correlation(image):
    # plt.imshow(image)
    # plt.show()
    dx = 3
    tmp = np.where(image==1)
    if tmp[0].shape[0] > 1:
        print("yeet")
    if tmp[0].shape[0] < 1:
        return 0,0,0
    imax = tmp[0][0]
    jmax = tmp[1][0]
    tofit = image[max(imax-dx,0):min(imax+dx+1,image.shape[0]),max(jmax-dx,0):min(jmax+dx+1,image.shape[1])].copy()

    image[max(imax-dx,0):min(imax+dx+1,image.shape[0]),max(jmax-dx,0):min(jmax+dx+1,image.shape[1])] = 0

    if (tofit.shape[0]<(2*dx+1)) or (tofit.shape[1]<(2*dx+1)):
        # fallback to not subpixel because i am lazy atm
        print("Yeet")
    else:
        imax,jmax = fit_gaussian(tofit, imax,jmax,dx)

    # imax,jmax = fit_gaussian(tofit, imax,jmax,dx)
    # plt.imshow(image)
    # plt.show()
    if np.max(image) == 1:
        print("Yikes")
    return imax,jmax,1/np.max(image)

def load_image_to_array(file):
    im = Image.open(file)
    return np.array(im)

def splitimage(im):
    return im[:im.shape[0]//2],im[im.shape[0]//2:]

def window(image,i,j,size):
    image = np.pad(image,size, mode="wrap")
    i = i+size
    j = j+size
    w = image[i:(i+size),j:(j+size)]
    return w - np.average(w)

def normalize(window):
    window = window.astype(np.float32)
    window -= window.mean()
    std = window.std()

    return np.clip(window,-5*std,5*std)

def gen_mask_array(image, x, y):
    img = Image.new('1', (image.shape[1], image.shape[0]),  0)
    polygon = np.hstack((x,y))
    polygon = polygon.flatten()
    polygon = list(np.array(polygon,dtype=np.int32))
    img1 = ImageDraw.Draw(img)
    img1.polygon(polygon, fill =1, outline =1)
    img1.line(polygon, fill=1, width=10)
    mask = np.array(img)
    return mask

class PIV_data():
    def __init__(self, xloc, yloc, u, v, s2n, filtr, image):
        self.x  = xloc
        self.y = yloc
        self.u = u
        self.v = v
        self.sn = s2n
        self.fil = filtr
        self.image = image

    def drawimage(self):
        plt.pcolormesh(np.linspace(0,np.max(self.x),self.image.shape[0]),np.linspace(0,np.max(self.y),self.image.shape[1]), self.image,shading='gouraud')


    def vector_speed_plot(self):
        # self.drawimage()
        cs = plt.contourf(self.x,self.y, np.sqrt(self.u**2+self.v**2))
        plt.quiver(self.x[self.fil],self.y[self.fil],self.u[self.fil],self.v[self.fil],angles='xy')
        plt.axis("equal")
        plt.colorbar(cs)
        plt.show()

    def u_vel_plot(self):
        # self.drawimage()
        cs = plt.contourf(self.x,self.y, self.u,25)
        plt.axis("equal")
        plt.colorbar(cs)
        plt.show()


    def __add__(self,other):
        u = self.u+other.u
        v = self.v+other.v
        fil = self.fil&other.fil
        u[np.invert(fil)] = np.NAN
        v[np.invert(fil)] = np.NAN

        return PIV_data(self.x, self.y,  u,v,self.sn,fil,self.image)

    def divide(self, scalar):
        self.u /= scalar
        self.v /= scalar

def RANS(sequence):
    u = np.zeros((sequence[0].u.shape[0],sequence[0].u.shape[1],len(sequence)))
    v = np.zeros_like(u)
    fil = np.zeros_like(sequence[0].fil)

    for i,s in enumerate(sequence):
        u[:,:,i] = s.u
        v[:,:,i] = s.v
        fil = fil | s.fil

    x,y,sn,im = sequence[0].x,sequence[0].y,sequence[0].sn,sequence[0].image
    avg =   PIV_data(x,y,np.nanmean(u,axis=2),np.nanmean(v,axis=2),sn,fil,im)
    std =   PIV_data(x,y,np.nanstd(u,axis=2),np.nanmean(v,axis=2),sn,fil,im)

    return avg,std




class Sequence():

    def __init__(self, folder, mask, cali, dt=0.0001):
        """
        Arguments:
        images:     file path to image files
        """
        images = os.listdir(folder)
        images = [f for f in images if f[-1]=="f"]
        self.images=images
        for i in range(len(self.images)):
            self.images[i] = folder+self.images[i]
        mask = scipy.io.loadmat(mask)
        self.xmask = mask["xmask"]+1
        self.ymask = mask["ymask"]+1
        self.mask = None
        self.cali = load_image_to_array(cali)
        self.dt = dt

        image1,image2 = splitimage(load_image_to_array(self.images[0]))
        self.avg = (image1+image2)/2/len(self.images)
        for i in range(1,len(self.images)):
            image1,image2 = splitimage(load_image_to_array(self.images[i]))
            self.avg += (image1+image2)/2/len(self.images)
        plt.imshow(self.cali)
        plt.show()


    def init_optical(self, M, pmm):
        raise NotImplementedError
    def get_length(self):
        return len(self.images)

    def PIV(self, N, size=64, num=0, s2n_cut=2, vavg=8):
        # image1 = load_image_to_array(self.images[num])
        # image2 = load_image_to_array(self.images[num+1])
        image1,image2 = splitimage(load_image_to_array(self.images[num]))
        if self.mask is None:
            self.mask = gen_mask_array(image1, self.xmask, self.ymask)

        image2 = image2 - self.avg#np.average(image2)
        image1 = image1 - self.avg
        mpp = 0.16732/image1.shape[1]
        # image2 = normalize(image2)
        # image1 = normalize(image1)
        image1[self.mask] = 0
        image2[self.mask] = 0

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(image1)
        # ax[1].imshow(image2)
        # ax[0].plot(self.xmask,self.ymask)
        # plt.show()
        im,jm = image1.shape
        if im > jm:
            Nx = int(N*im/jm)
            Ny = N
        else:
            Ny = int(N*jm/im)
            Nx = N

        ii = np.linspace(size,im-size,Nx,dtype=np.int64)
        jj = np.linspace(size,jm-size,Ny,dtype=np.int64)
        ii = np.linspace(0,im-1,Nx,dtype=np.int64)
        jj = np.linspace(0,jm-1,Ny,dtype=np.int64)

        ii,jj = np.meshgrid(ii,jj)
        di,dj = np.zeros_like(ii,dtype=np.float64),np.zeros_like(ii,dtype=np.float64)
        sn = np.zeros_like(ii,dtype=np.float64)
        ic, jc = [], []
        fil = np.zeros_like(ii)
        c = 0
        subsize = 2*vavg+size

        def correlator(i,j):
            if self.mask[ii[i,j],jj[i,j]]:
                return
            window1 = window(image1,ii[i,j]-size//2,jj[i,j]-size//2,size)
            window2 = window(image2,ii[i,j]-subsize//2,jj[i,j]-subsize//2,subsize)
            corr = signal.correlate(window2, window1, mode="same", method="fft")
            corr = corr/np.max(corr)
            ipos,jpos,s2n = process_correlation(corr)
            di[i,j] = ipos-ii[i,j]+ii[i,j]-subsize//2
            dj[i,j] = jpos-jj[i,j]+jj[i,j]-subsize//2
            sn[i,j] = s2n
            return
        # a_pool = multiprocessing.Pool()

        for i in range(ii.shape[0]):
            for j in range(ii.shape[1]):
                correlator(i,j)
                # if self.mask[ii[i,j],jj[i,j]]:
                #     c += 1
                #     continue
                # window1 = window(image1,ii[i,j]-size//2,jj[i,j]-size//2,size)
                # corr = signal.correlate(image2, window1, mode="same", method="fft")
                # corr = corr/np.max(corr)
                # ipos,jpos,s2n = process_correlation(corr)
                # print(s2n)
                # di[i,j] = ipos-ii[i,j]
                # dj[i,j] = jpos-jj[i,j]
                # sn[i,j] = s2n
        print(s2n_cut)
        ii = np.array(ii,dtype=np.float64)
        jj = np.array(jj,dtype=np.float64)

        ii *= mpp
        jj *= mpp
        print(np.average(di))
        print(np.average(dj))

        di *= mpp/self.dt
        dj *= mpp/self.dt
        umag = np.sqrt(dj**2+di**2)
        fil = (sn >= s2n_cut) & (umag < 13)
        
        print(c)
        print(np.sum(fil)/ii.shape[0]/ii.shape[1])
        print(sn[fil])
        dj[np.invert(fil)] = np.NAN
        di[np.invert(fil)] = np.NAN


        return PIV_data(jj,ii,dj,di,sn,fil, image1)


        fig, ax = plt.subplots(1,2)
        # ax[0].imshow(image1)
        ax[0].pcolormesh(jj,ii, np.sqrt(dj**2+di**2),shading='gouraud')
        cs=ax[0].quiver(jj[fil],ii[fil],dj[fil],di[fil],angles='xy')
        # plt.scatter(jj[fil],ii[fil],c="r",marker="x")
        ax[1].boxplot(sn[fil].flatten())
        ax[1].boxplot(np.sqrt(dj**2+di**2)[fil].flatten())
        # ax[0].set_ylim(ax[0].get_ylim()[::-1])
        ax[0].imshow(self.mask, cmap="spring")
        plt.show()
        


Sq = Sequence("../PIV_data/Alpha15_dt100/","../PIV_matlab_codes/WIDIM/Mask_Alpha_15.mat", "../PIV_data/Calibration/B00001.tif")
l = 25#Sq.get_length()
N = 150
Sn = 1.0
s = 64
call = lambda x: Sq.PIV(N,s2n_cut=Sn,size=s, num=x)
call(99).u_vel_plot()
# res = []
# for i in range(0,l):
#     res.append(call(i))

# avg,rms = RANS(res)
# avg.u_vel_plot()
# pickle.dump(avg, open("avg2.pckl","wb"))
# pickle.dump(rms, open("rms2.pckl","wb"))
avg = pickle.load(open("avg.pckl","rb"))
avg.u_vel_plot()

