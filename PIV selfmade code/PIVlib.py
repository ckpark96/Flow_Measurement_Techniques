import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats

def fit_gaussian(image,ic,jc,dx):
    tmp = np.where(image==1)
    
    imax = tmp[0][0]
    jmax = tmp[1][0]

    return imax+(ic-dx),jmax+(jc-dx)

def process_correlation(image):
    # plt.imshow(image)
    # plt.show()
    dx = 45
    tmp = np.where(image==1)
    if tmp[0].shape[0] > 1:
        return 1
    imax = tmp[0][0]
    jmax = tmp[1][0]
    tofit = image[max(imax-dx,0):min(imax+dx+1,image.shape[0]),max(jmax-dx,0):min(jmax+dx+1,image.shape[1])].copy()
    image[max(imax-dx,0):min(imax+dx+1,image.shape[0]),max(jmax-dx,0):min(jmax+dx+1,image.shape[1])] = 0

    # imax,jmax = fit_gaussian(tofit, imax,jmax,dx)

    return imax,jmax,1/np.max(image)

def load_image_to_array(file):
    im = Image.open(file)
    return np.array(im)

def splitimage(im):
    return im[:im.shape[0]//2],im[im.shape[0]//2:]

def window(image,i,j,size):
    w = image[i:i+size,j:j+size]
    return w - np.average(w)


class Sequence():

    def __init__(self, images, dt=0.01):
        """
        Arguments:
        images:     file path to image files
        """
        self.images=images

    def init_optical(self, M, pmm):
        raise NotImplementedError

    def PIV(self,N,size=64,num=0,s2n_cut=2):
        # image1 = load_image_to_array(self.images[num])
        # image2 = load_image_to_array(self.images[num+1])
        image1,image2 = splitimage(load_image_to_array(self.images[num]))
        image2 = image2 - np.average(image2)
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(image1)
        ax[1].imshow(image2)
        plt.show()
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
        di,dj = np.zeros_like(ii),np.zeros_like(ii)
        sn = np.zeros_like(ii)

        for i in range(ii.shape[0]):
            for j in range(ii.shape[1]):
                print(j)
                window1 = window(image1,ii[i,j]-size//2,jj[i,j]-size//2,size)
                corr = signal.correlate(image2, window1, mode="same", method="fft")
                corr = corr/np.max(corr)
                ipos,jpos,s2n = process_correlation(corr)
                di[i,j] = ipos-ii[i,j]
                dj[i,j] = jpos-jj[i,j]
                sn[i,j] = s2n
        fil = sn >= s2n_cut
                
        print(np.sum(fil))
        plt.hist(sn)
        plt.show()
        # plt.imshow(image1)
        plt.quiver(jj[fil],ii[fil],dj[fil],di[fil],angles='xy')
        plt.show()
        


Sq = Sequence(['../PIV_data/Alpha0_dt1/B00001.tif'])
Sq.PIV(30,s2n_cut=1,size=64)
