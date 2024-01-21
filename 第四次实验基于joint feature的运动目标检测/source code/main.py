import os
import cv2
import numpy as np
import joint_feature_spatial as JFS


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
    video_path = "C:\\Users\\WYF\\Desktop\DIP\\object_tracking\\768x576.avi"
    video_savepath = "C:\\Users\\WYF\\Desktop\\DIP\\object_tracking\\Result2.mp4"

    frames = readVideo(video_path)
    save_frames = []

    #(x, y, w, h) = JFS.getFirstbox(frames)    
    
    #init_pos = np.array([305, 478])
    init_pos = np.array([280, 665])
    newpos = init_pos
    test_frame = frames[0].copy()

    h=70
    w=30
    #cv2.rectangle(test_frame, (int(newpos[1])-30//2, int(newpos[0])-70//2), (int(newpos[1])+30//2, int(newpos[0])+70//2), (0, 0, 255), 2)
    cv2.rectangle(test_frame, (int(newpos[1])-w//2, int(newpos[0])-h//2), (int(newpos[1])+w//2, int(newpos[0])+h//2), (0, 0, 255), 2)
    cv2.imshow("test", test_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_frames.append(test_frame)
    #print(len(frames))

    for i in range(1, len(frames)-100):
        new_frame = frames[i].copy()
        newpos, EDGE = JFS.mspos(initimg=frames[0], newimg=new_frame, h=h, w=w, initpos=init_pos, oldpos=newpos, epsilon=0.5, maxits=10)
        newpos = np.int32(newpos)

        if (i%20==0):
            cv2.imshow("edge",  EDGE[int(newpos[0]-h//2):int(newpos[0]+h//2),int(newpos[1]-w//2):int(newpos[1]+w//2)])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow("rgb",  new_frame[int(newpos[0]-h//2):int(newpos[0]+h//2),int(newpos[1]-w//2):int(newpos[1]+w//2)])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
            cv2.rectangle(new_frame, (int(newpos[1])-w//2, int(newpos[0])-h//2), (int(newpos[1])+w//2, int(newpos[0])+h//2), (0, 0, 255), 2)
            cv2.imshow("test", new_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #cv2.rectangle(new_frame, (int(newpos[1])-w//2, int(newpos[0])-h//2), (int(newpos[1])+w//2, int(newpos[0])+h//2), (0, 0, 255), 2)
        #save_frames.append(new_frame)

    #writeVideo(save_frames, video_savepath, fps=20)

        



