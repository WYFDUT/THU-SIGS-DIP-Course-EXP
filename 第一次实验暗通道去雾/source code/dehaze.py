import cv2
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


class ImgDehaze:
    def __init__(self) -> None:
        pass

    def DarkChannel(self, I, r):
        """
        :param I: original image 
        :param r: local area radius(mininum filter radius)
        :return: dark channel image
        """
        I = np.array(I)
        h, w, c = I.shape

        # each pixel find min channel value
        I = I.min(axis=2)

        # create a rectangle kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*r+1, 2*r+1))

        # padding the image for next operations
        I_padding = np.zeros(shape=(h+2*r, w+2*r))*1.0
        I_padding[r:h+r, r:r+w] = I
        I_dark_padding = np.zeros_like(I_padding)

        # loop each pixel
        # or use erode operation cv2.erode(I, kernel)
        # cv2.erode(I, kernel)
        for i in range(r, r+h):
            for j in range(r, r+w):
                local_area = I_padding[(i-r):(i+r+1), (j-r):(j+r+1)]
                local_min = np.min(local_area[kernel==1])
                I_dark_padding[i,j] = local_min
        
        # depadding operations
        I_dark = I_dark_padding[r:r+h, r:r+w]
        return I_dark
    
    def DarkChannelInverse(self, I, r):
        # A test function for replacing the order of the minima filter
        """
        :param I: original image 
        :param r: local area radius(mininum filter radius)
        :return: dark channel image
        """
        I = np.array(I)
        h, w, c = I.shape

        # create a rectangle kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*r+1, 2*r+1))

        # padding the image for next operations
        I_padding = np.zeros(shape=(h+2*r, w+2*r, 3))*1.0
        I_padding[r:h+r, r:r+w, 0] = I[:,:,0]
        I_padding[r:h+r, r:r+w, 1] = I[:,:,1]
        I_padding[r:h+r, r:r+w, 2] = I[:,:,2]
        I_dark_padding = np.zeros_like(I_padding)

        # loop each pixel
        # or use erode operation cv2.erode(I, kernel)
        # cv2.erode(I, kernel)
        for k in range(c):
            for i in range(r, r+h):
                for j in range(r, r+w):
                    local_area = I_padding[(i-r):(i+r+1), (j-r):(j+r+1), k]
                    local_min = np.min(local_area[kernel==1])
                    I_dark_padding[i,j,k] = local_min
        
        # depadding operations
        I_dark = I_dark_padding[r:r+h, r:r+w, :]

        # each pixel find min channel value
        I_dark = I_dark.min(axis=2)

        return I_dark

    def AtmLight(self, I, dark):
        """
        :param I: original image 
        :param dark: dark channel image 
        :return: each channel atmosphere intensity
        """
        h,w,c = I.shape
        imsz = h*w
        # definea a number that 0.1% brightest pixels
        numpx = int(max(math.floor(imsz/1000),1))
        darkvec = dark.reshape(imsz)
        imvec = I.reshape(imsz,3)

        # pick the top 0.1% brightest pixels in the dark channel and return the index
        indices = darkvec.argsort()
        indices = indices[imsz-numpx:]

        A = np.zeros([1,3])
        for ind in range(1,numpx):
        # the pixels with highest intensity in the input image I is selected as the atmospheric light.
        # other method use normalized intensity in the input image I as the atmospheric light.
        # atmsum = atmsum + imvec[indices[ind]]
            for i in range(0, 3):
                A[0, i] = max(A[0, i], imvec[indices[ind]][i])
        #A = atmsum / numpx
        return A

    def TransmissionEstimate(self,I,A,r,omega=0.95):
        """
        :param I: original image 
        :param A: atmospheric light
        :param r: local area radius
        :return: the estimated transmission
        """
        im3 = np.empty(I.shape,I.dtype)

        # each channel should norm
        for ind in range(0,3):
            im3[:,:,ind] = I[:,:,ind]/A[0,ind]

        transmission = 1 - omega*self.DarkChannel(im3,r)
        return transmission

    def Guidedfilter(self,I,p,r,eps):
        """
        :param I: original gray image 
        :param p: original transmission map
        :param r: filter radius default is 60
        :param eps: linear regression parameter default is 0.0001
        :return optimilized transmission map
        """
        # in the article, author regard a=cov(I,p)/(var(I,p)+eps), b=mean(p)-a*mean(I)
        # cauculate cov(I,p) first
        mean_I = cv2.boxFilter(I,cv2.CV_64F,(r,r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
        mean_Ip = cv2.boxFilter(I*p,cv2.CV_64F,(r,r))
        cov_Ip = mean_Ip - mean_I*mean_p

        # cauculate var(I) next
        mean_II = cv2.boxFilter(I*I,cv2.CV_64F,(r,r))
        var_I   = mean_II - mean_I*mean_I

        # cauculate a & b finally
        a = cov_Ip/(var_I + eps)
        b = mean_p - a*mean_I

        mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
        mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

        q = mean_a*I + mean_b
        return q

    def TransmissionRefine(self,I,et):
        """
        :param I: original image 
        :return: optimilized transmission map
        """
        # convert BGR image to gray
        gray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray)/255
        r = 60
        eps = 0.0001

        # guidedfilter for transmission map
        t = self.Guidedfilter(gray,et,r,eps)

        return t

    def Softmatting(self, transmission, image, epsilon=1e-6, window_radius=1):
        h, w = transmission.shape
        N = h * w
        window_size = (2 * window_radius + 1) ** 2

        transmission_flat = transmission.flatten()
        image_flat = image.reshape((N, 3))

        A = sparse.lil_matrix((N, N))
        b = np.zeros(N)

        for y in range(h):
            for x in range(w):
                idx = y * w + x
                window_indices = []

                for j in range(max(0, y - window_radius), min(h, y + window_radius + 1)):
                    for i in range(max(0, x - window_radius), min(w, x + window_radius + 1)):
                        window_indices.append(j * w + i)

                for i in window_indices:
                    A[idx, i] = np.dot(image_flat[idx], image_flat[i])
                A[idx, idx] += epsilon
                b[idx] = np.dot(image_flat[idx], image_flat[idx])

        transmission_smooth = spsolve(A.tocsr(), b)
        transmission_smooth = np.clip(transmission_smooth, 0, 1)
        transmission_smooth = transmission_smooth.reshape((h, w))

        return transmission_smooth

    def Recover(self,I,t,A,tx = 0.1):
        """
        :param I: original image 
        :param t: the estimated transmission
        :param A: each channel atmosphere intensity
        :return: dehazed figure
        """
        res = np.empty(I.shape,I.dtype)
        t = cv2.max(t,tx)

        for ind in range(0,3):
            res[:,:,ind] = (I[:,:,ind]-A[0,ind])/t + A[0,ind]
        return res

def png2jpg():
    # a simple func for changging image format 
    path = os.path.abspath('.')
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == '.png': 
            img = cv2.imread(os.path.join(path, filename))
            newfilename = filename.replace(".png", ".jpg")
            cv2.imwrite(os.path.join(path, newfilename), img)  


if __name__ == '__main__':
    # png2jpg()

    src = cv2.imread("rainy1.jpg")

    I = src.astype('float64')/255
    
    id = ImgDehaze()
    start_time = time.time()
    dark = id.DarkChannel(I,2)
    end_time = time.time()
    print("dark channel cauculate time is {} s".format(end_time-start_time))
    
    A = id.AtmLight(I,dark)

    t = id.TransmissionEstimate(I,A,20)

    start_time = time.time()
    t = id.TransmissionRefine(src,t)
    end_time = time.time()
    print("optimization time is {} s".format(end_time-start_time))

    J = id.Recover(I,t,A,0.1)
    # print(J.dtype, dark.dtype)

    cv2.imshow("dark",dark)
    cv2.imshow("t",t)
    cv2.imshow('I',src)
    cv2.imshow('J',J)
    cv2.imwrite("rainy1res.png",J*255)
    cv2.waitKey()
    cv2.destroyAllWindows()  


