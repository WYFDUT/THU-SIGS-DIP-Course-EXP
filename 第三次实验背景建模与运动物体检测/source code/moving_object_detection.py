import os 
import cv2
import math
import time 
import tqdm
import fnmatch
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pylab as plt
from collections import defaultdict


def frame_diff(frames, background = None):
    new_frames = []
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
    for index, frame in enumerate(frames):
        new_frame = frame.copy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if background is None:
            background = gray_frame
            continue

        diff = cv2.absdiff(cv2.GaussianBlur(cv2.cvtColor(frames[index-1], cv2.COLOR_BGR2GRAY), (21, 21), 0), gray_frame)
        diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] # OSTU method
        diff = cv2.dilate(diff, es, iterations=2) # Dilation
        '''
        cv2.imshow("test", diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        # Mark the moving object
        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Detect the contours
        for c in contours:
            if cv2.contourArea(c) < 600: # Param 600 is acquired by experiments
                continue
            (x, y, w, h) = cv2.boundingRect(c) # Cauculate the bounding box
            cv2.rectangle(new_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.imshow("test", new_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        new_frames.append(new_frame)
        cv2.destroyAllWindows()    
    return new_frames


def single_gaussian_modeling(frames, alpha=0.03, stdInit=20.0, lamda=2.5*1.2):
    # Initialize the background & foreground
    frame_u = (frames[0].copy().astype("float64")+frames[9].copy().astype("float64")+frames[19].copy().astype("float64"))/3
    frame_d = np.zeros_like(frames[0], dtype="float64")
    frame_std = np.ones_like(frames[0], dtype="float64")*stdInit
    frame_var = np.ones_like(frames[0], dtype="float64")*stdInit*stdInit
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
    new_frames = []

    for index in range(len(frames)):
        new_frame = frames[index].copy()
        frame_cur = frames[index].copy().astype("float64")
        for h in range(frames[0].shape[0]):
            for w in range(frames[0].shape[1]):
                # Judge the pixel belong to background or foreground
                if ((abs(frame_cur[h, w, 0]-frame_u[h, w, 0])<lamda*frame_std[h, w, 0]) and (
                    abs(frame_cur[h, w, 1]-frame_u[h, w, 1])<lamda*frame_std[h, w, 1]) and (
                    abs(frame_cur[h, w, 2]-frame_u[h, w, 2])<lamda*frame_std[h, w, 2])):
                    # Update the background
                    frame_u[h, w, :] = (alpha)*frame_u[h, w, :] + (1-alpha)*frame_cur[h, w, :]
                    frame_var[h, w, :] = (alpha)*frame_var[h, w, :] + (1-alpha)*(frame_cur[h, w, :]-frame_u[h, w, :])**2
                    frame_std[h, w, :] = np.sqrt(frame_var[h, w, :])
                #else:
                    # Update the foreground
                    
        frame_d[:, :, :] = np.clip(abs(frame_u[:, :, :]-frame_cur[:, :, :]), 0, 255)
        '''
        cv2.imshow("test1", cv2.threshold(cv2.cvtColor(np.uint8(frame_d), cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cv2.imshow("test2", np.uint8(frame_u-frames[0]))
        cv2.imshow("test2", np.uint8(frame_u))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        diff = cv2.threshold(cv2.cvtColor(np.uint8(frame_d), cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] # OSTU method
        # Erode & Dilate
        diff = cv2.erode(diff, es, iterations=1)
        diff = cv2.dilate(diff, es, iterations=1)
        # Mark the moving object
        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Detect the contours
        for c in contours:
            if cv2.contourArea(c) < 300: # Param 300 is acquired by experiments
                continue
            (x, y, w, h) = cv2.boundingRect(c) # Cauculate the bounding box
            cv2.rectangle(new_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        new_frames.append(new_frame)
        '''
        cv2.imshow("test3", new_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
    return new_frames

def multiple_gaussian_modeling(frames):
    new_frames = []
    # Parameters
    C = 3  # Number of gaussian components
    M = 3  # Number of background components
    sd_init = 6  # Initial standard deviation for new components
    alph = 0.01  # Learning rate
    D = 2.5  # Positive deviation threshold
    thresh = 0.25  # Foreground threshold
    p = alph / (1 / C)  # Initial p variable
    height, width, channel = frames[0].shape

    frg = np.zeros((height, width, channel), dtype="float64")  # Foreground array
    bg_bw = np.zeros((height, width, channel), dtype="float64")  # Background array

    # Initialize arrays for weights, means, standard deviations, and differences
    w = np.full((height, width, C, channel), 1 / C, dtype="float64")
    mean = np.random.rand(height, width, C, channel) * 255
    sd = np.full((height, width, C, channel), sd_init, dtype="float64")
    u_diff = np.zeros((height, width, C, channel), dtype="float64")
    rank = np.zeros((C, channel), dtype="float64")

    for mframe in frames:
        new_frame = mframe.copy()
        for index in range(channel):
            current = mframe[:, :, index]
            rank_ind = np.zeros((C), dtype="uint8")

            # Calculate difference of pixel values from mean
            for m in range(C):
                u_diff[:, :, m, index] = np.abs((current[:, :]) - mean[:, :, m, index])
            # Update Gaussian components for each pixel
            for i in range(height):
                for j in range(width):
                    match = 0
                    temp = 0
                    for k in range(C):
                        if abs(u_diff[i, j, k, index]) <= D * sd[i, j, k, index]:
                            match = 1
                            w[i, j, k, index] = (1 - alph) * w[i, j, k, index] + alph
                            p = alph / w[i, j, k, index]
                            mean[i, j, k, index] = (1 - p) * mean[i, j, k, index] + p * current[i, j]
                            sd[i, j, k, index] = (1 - p) * sd[i, j, k, index] + p * (current[i, j] - mean[i, j, k, index]) ** 2
                        else:
                            w[i, j, k, index] = (1 - alph) * w[i, j, k, index]
                        temp += w[i, j, k, index]
                    for k in range(C):
                        w[i, j, k, index] /= temp
                    temp = w[i, j, 0, index]
                    bg_bw[i, j, index] = 0
                    for k in range(C):
                        bg_bw[i, j, index] += mean[i, j, k, index] * w[i, j, k, index]
                        if w[i, j, k, index] <= temp:
                            min_index = k
                            temp = w[i, j, k, index]
                        rank_ind[k] = k
                    # Update the foreground image
                    if match == 0:
                        mean[i, j, min_index, index] = current[i, j]
                        sd[i, j, min_index, index] = sd_init
                    for k in range(C):
                        rank[k, index] = w[i, j, k, index]/sd[i, j, k, index]
                    # sort rank values
                    for k in range(C):
                        for m in range(k):
                            if rank[k, index] > rank[m, index]:   # swap
                                rank[k, index], rank[m, index] = rank[m, index], rank[k, index]
                                rank_ind[k], rank_ind[m] = rank_ind[m], rank_ind[k]
                    # Calculate foreground
                    match = 0
                    k = 0
                    while match == 0 and k < M:
                        if w[i, j, rank_ind[k], index] >= thresh:
                            if abs(u_diff[i, j, rank_ind[k], index]) <= D * sd[i, j, rank_ind[k], index]:
                                frg[i, j, index] = 0
                                match = 1
                            else:
                                frg[i, j, index] = current[i, j]
                        k += 1
    
        diff = cv2.cvtColor(np.uint8(frg), cv2.COLOR_BGR2GRAY)
        '''
        cv2.imshow("test", diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("bg", np.uint8(bg_bw))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Detect the contours
        for c in contours:
            if cv2.contourArea(c) < 600: # Param 600 is acquired by experiments
                continue
            (x0, y0, w0, h0) = cv2.boundingRect(c) # Cauculate the bounding box
            cv2.rectangle(new_frame, (x0, y0), (x0+w0, y0+h0), (0, 0, 255), 2)
        '''
        cv2.imshow("test", new_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        new_frames.append(new_frame)
    return new_frames

#######################
    """
    # GMM model
    new_frames = []
    height, width, channel = frames[0].shape
    fgbg = cv2.createBackgroundSubtractorMOG2()
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
    bg = np.random.rand(height, width, channel) * 255
    mask = np.zeros_like(frames[0], dtype="float64")

    for i in range(len(frames)):
        new_frame = frames[i].copy()
        fgmask = fgbg.apply(frames[i])

        diff = fgmask
        diff = cv2.erode(diff, es, iterations=1)
        diff = cv2.dilate(diff, es, iterations=1)
        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Detect the contours
        for c in contours:
            if cv2.contourArea(c) < 600: # Param 600 is acquired by experiments
                continue
            (x, y, w, h) = cv2.boundingRect(c) # Cauculate the bounding box
            cv2.rectangle(new_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        new_frames.append(new_frame)

    return new_frames
    """
#######################