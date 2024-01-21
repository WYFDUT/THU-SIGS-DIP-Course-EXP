import cv2
import numpy as np
from matplotlib import pyplot as plt


def getFirstbox(frames):
    new_frame = frames[1].copy()
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
    gray_frame0, gray_frame1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(frames[2], cv2.COLOR_BGR2GRAY)
    gray_frame0, gray_frame1 = cv2.GaussianBlur(gray_frame0, (21, 21), 0), cv2.GaussianBlur(gray_frame1, (21, 21), 0)

    diff = cv2.absdiff(gray_frame0, gray_frame1)
    diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] # OSTU method
    diff = cv2.erode(diff, es, iterations=1)
    diff = cv2.dilate(diff, es, iterations=1)

    # Mark the moving object
    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Detect the contours
    for c in contours:
        if cv2.contourArea(c) < 800:
            continue
        (x, y, w, h) = cv2.boundingRect(c) # Cauculate the bounding box
    return (x-20, y-2, w, h)


########## Joint Feature Spatial #############
def inddisk(h, w, pos):
    index = np.zeros([2, h, w])
    index[0] = np.repeat(np.arange(pos[0]-h//2, pos[0]+h//2, 1).reshape(1, -1), w, axis=0).transpose()
    index[1] = np.repeat(np.arange(pos[1]-w//2, pos[1]+w//2, 1).reshape(1, -1), h, axis=0)
    return index


def kernel(x, y, sigma):
	y=np.array(y)
	x=np.array(x)
	x=x.reshape(-1,1,1)
	return np.exp(-1*np.sum((x-y)**2,axis=0)/(2*(sigma**2)))


def mspos(initimg, newimg, h, w, initpos, oldpos, epsilon, maxits, sigma=20, k_h=10):
    y_center=oldpos
    x_index = inddisk(h, w, initpos)
    omega=kernel(initpos, x_index, sigma)

    EDGE1 = np.zeros_like(initimg)
    for i in range(EDGE1.shape[2]):
        grad_x = cv2.convertScaleAbs(cv2.Sobel(initimg[:, :, i], cv2.CV_16S, 1, 0))
        grad_y = cv2.convertScaleAbs(cv2.Sobel(initimg[:, :, i], cv2.CV_16S, 0, 1))
        EDGE1[:, :, i] = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    
    EDGE = np.zeros_like(newimg)
    for i in range(EDGE.shape[2]):
        grad_x = cv2.convertScaleAbs(cv2.Sobel(newimg[:, :, i], cv2.CV_16S, 1, 0))
        grad_y = cv2.convertScaleAbs(cv2.Sobel(newimg[:, :, i], cv2.CV_16S, 0, 1))
        EDGE[:, :, i] = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    for k in range(maxits):
        y_feat=EDGE[int(y_center[0]-h//2):int(y_center[0]+h//2),int(y_center[1]-w//2):int(y_center[1]+w//2)].transpose(2,0,1)
        y_RGB=newimg[int(y_center[0]-h//2):int(y_center[0]+h//2),int(y_center[1]-w//2):int(y_center[1]+w//2)].transpose(2,0,1)
        
        y_feature=np.concatenate((y_feat, y_RGB),axis=0)
        #y_feature = y_RGB

        sumxyuv = 0.0
        sumyxyuv = np.zeros_like(y_center).astype("float64")
        y_index = inddisk(h,w,y_center)
        g=kernel(y_center, y_index, sigma)
        for i in range(h):
            for j in range(w):
                x1=initpos[0]-h//2+i
                x2=initpos[1]-w//2+j

                x_feature = np.concatenate((EDGE1[x1,x2], initimg[x1,x2]))
                #x_feature = initimg[x1,x2]
                ki=kernel(x_feature,y_feature, k_h)
                
                sumyxyuv += np.sum(np.sum(g*ki*y_index,axis=1),axis=1)*omega[i,j]
                sumxyuv += np.sum(g*ki)*omega[i,j]
        y_new=sumyxyuv / sumxyuv
        if np.linalg.norm(y_new - y_center) < epsilon:
            break
        y_center=np.round(y_new)
    print("Iteration rounds: {}".format(min(k+1, maxits)))
    return y_center, EDGE
    

    




