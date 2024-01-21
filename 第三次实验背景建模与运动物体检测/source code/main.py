import os
import cv2
import numpy as np
import moving_object_detection


def readVideo(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False 

    while ret:
        ret, frame  = cap.read()
        if ret == None:
            break
        frames.append(frame)
    frames.pop(-1)
    return frames

def writeVideo(frames, video_savepath, fps=10):
    image_height, image_width = frames[0].shape[0], frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_write = cv2.VideoWriter(video_savepath, fourcc, fps, (image_width, image_height))
    for i in range(len(frames)):
        video_write.write(frames[i])
    video_write.release()


if __name__ == "__main__":
    video_path = "C:\\Users\\WYF\\Desktop\\DIP\\GMM_background_modeling\\768x576.avi"
    video_savepath = "C:\\Users\\WYF\\Desktop\\DIP\\GMM_background_modeling"

    frames = readVideo(video_path)

    #multiple_gauss_frames = moving_object_detection.multiple_gaussian_modeling(frames)

    single_gauss_frames = moving_object_detection.single_gaussian_modeling(frames)

    #diff_frames = moving_object_detection.frame_diff(frames)

    writeVideo(single_gauss_frames, os.path.join(video_savepath, ('SingleGaussian'+'.mp4')))



