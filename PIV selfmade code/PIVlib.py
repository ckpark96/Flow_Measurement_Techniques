import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import scipy
import scipy.io
import os
import multiprocessing

"""
TODO: Loop Parallelization
TODO: Proper scaling of values to velocity (May not even be feasible)
TODO: Subpixel interpolation
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
    dx = 10
    tmp = np.where(image==1)
    if tmp[0].shape[0] > 1:
        print("yeet")
    if tmp[0].shape[0] < 1:
        return 0,0,0
    imax = tmp[0][0]
    jmax = tmp[1][0]
    tofit = image[max(imax-dx,0):min(imax+dx+1,image.shape[0]),max(jmax-dx,0):min(jmax+dx+1,image.shape[1])].copy()

    image[max(imax-dx,0):min(imax+dx+1,image.shape[0]),max(jmax-dx,0):min(jmax+dx+1,image.shape[1])] = 0

    if (tofit.shape[0]<21) or (tofit.shape[1]<21):
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
    w = image[i:i+size,j:j+size]
    return w - np.average(w)

def normalize(window):
    window = window.astype(np.float32)
    window -= window.mean()
    std = window.std()

    return np.clip(window,0,5*std)

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
class Sequence():

    def __init__(self, folder, mask, cali, dt=0.01):
        """
        Arguments:
        images:     file path to image files
        """
        images = os.listdir(folder)
        self.images=images
        for i in range(len(self.images)):
            self.images[i] = folder+self.images[i]
        mask = scipy.io.loadmat(mask)
        self.xmask = mask["xmask"]+1
        self.ymask = mask["ymask"]+1
        self.mask = None
        self.cali = load_image_to_array(cali)

        image1,image2 = splitimage(load_image_to_array(self.images[0]))
        self.avg = (image1+image2)/2/len(self.images)
        for i in range(1,len(self.images)):
            image1,image2 = splitimage(load_image_to_array(self.images[i]))
            self.avg += (image1+image2)/2/len(self.images)
        print("")


    def init_optical(self, M, pmm):
        raise NotImplementedError

    def PIV(self, N, size=64, num=0, s2n_cut=2, bbi=[0,1e12],bbj=[0,1e12]):
        # image1 = load_image_to_array(self.images[num])
        # image2 = load_image_to_array(self.images[num+1])
        image1,image2 = splitimage(load_image_to_array(self.images[num]))
        if not self.mask:
            self.mask = gen_mask_array(image1, self.xmask, self.ymask)

        image2 = image2 - self.avg#np.average(image2)
        image1 = image1 - self.avg
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

        ii,jj = np.meshgrid(ii,jj)
        di,dj = np.zeros_like(ii,dtype=np.float64),np.zeros_like(ii,dtype=np.float64)
        sn = np.zeros_like(ii,dtype=np.float64)
        ic, jc = [], []
        fil = np.zeros_like(ii)
        c = 0

        def correlator(i,j):
            if self.mask[ii[i,j],jj[i,j]]:
                return
            window1 = window(image1,ii[i,j]-size//2,jj[i,j]-size//2,size)
            corr = signal.correlate(image2, window1, mode="same", method="fft")
            corr = corr/np.max(corr)
            ipos,jpos,s2n = process_correlation(corr)
            di[i,j] = ipos-ii[i,j]
            dj[i,j] = jpos-jj[i,j]
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
        fil = (sn >= s2n_cut)
        
        print(c)
        print(np.sum(fil)/ii.shape[0]/ii.shape[1])
        print(sn[fil])
        dj[np.invert(fil)] = np.NAN
        di[np.invert(fil)] = np.NAN
        fig, ax = plt.subplots(1,2)
        # ax[0].imshow(image1)
        ax[0].pcolormesh(jj,ii, np.sqrt(dj**2+di**2),shading='gouraud')
        cs=ax[0].quiver(jj[fil],ii[fil],dj[fil],di[fil],angles='xy')
        # plt.scatter(jj[fil],ii[fil],c="r",marker="x")
        ax[1].boxplot(sn[fil].flatten())
        ax[1].boxplot(np.sqrt(dj**2+di**2)[fil].flatten())
        ax[0].set_ylim(ax[0].get_ylim()[::-1])
        plt.show()
        


Sq = Sequence("../PIV_data/Alpha0_dt100/","../PIV_matlab_codes/WIDIM/Mask_Alpha_0.mat", "../PIV_data/Calibration/B00001.tif")
Sq.PIV(30,s2n_cut=1.5,size=128, num=7)
