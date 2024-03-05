from PIL import Image
from ultralytics import YOLO
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import predict_mbv2 as Classify
import bcp_dcp as dcp

"""
image_path = '/root/wyf/Violation_Detect/Vehicle_Detect/test9.png'
condition = Classify.judge_day_night(image_path)
if condition == "Night":
    print("Night")
    img = cv2.imread(image_path)
    img = dcp.enhance(img)
else:
    print("Day")
    img = cv2.imread(image_path)
    #img = dcp.enhance(img)
"""

def opical_judge(bboxs, frame1, frame2):

    # Motion Analysis
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    detect_res = []

    # Calculate speed or displacement for each bounding box
    for bbox in bboxs:
        x1, y1, x2, y2 = int(bbox[0]-bbox[2]//2), int(bbox[1]-bbox[3]//2), int(bbox[0]+bbox[2]//2), int(bbox[1]+bbox[3]//2) 

        # Calculate average flow within the bounding box
        avg_flow = np.mean(flow[y1:y2, x1:x2], axis=(0, 1))

        # Calculate speed or displacement
        speed = np.linalg.norm(avg_flow)

        # Set a threshold to determine stationary objects
        threshold = 2.5  # Adjust this threshold based on your scenario

        if speed < threshold:
            #print(" is stationary.")
            detect_res.append(bbox)
        else:
            #print(" is moving.")
            pass
    return detect_res


# 加载预训练的YOLOv8n模型
model = YOLO('/root/wyf/detection/runs/detect/train4/weights/best.pt')

# 打开视频文件
video_path = "/root/wyf/Violation_Detect/Vehicle_Detect/4_cut.mp4"
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
out = cv2.VideoWriter('track_detect.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame.shape[1], frame.shape[0]))  # 保存视频
tmp = np.zeros_like(frame)
flag = True

# 存储追踪历史
track_history = defaultdict(lambda: [])

# 循环遍历视频帧
while cap.isOpened():
    # 从视频读取一帧
    success, frame = cap.read()
    

    if success:
        #condition = Classify.judge_day_night(frame)
        #if condition == "Night":
            #frame = dcp.enhance(frame)
        #else:
            #pass
            #img = dcp.enhance(img)
        # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[3,4,5,8])

        # 获取框和追踪ID
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # 在帧上展示结果
        a = np.array(results[0].boxes.xywh.cpu())
        if flag == True:
            bboxs = opical_judge(a, tmp, frame)
            annotated_frame = frame.copy()
            for bbox in a:
                cv2.rectangle(annotated_frame, (int(bbox[0]-bbox[2]//2), int(bbox[1]-bbox[3]//2)), (int(bbox[0]+bbox[2]//2), int(bbox[1]+bbox[3]//2)), (0, 255, 0), 3)
            for bbox in bboxs:
                cv2.rectangle(annotated_frame, (int(bbox[0]-bbox[2]//2), int(bbox[1]-bbox[3]//2)), (int(bbox[0]+bbox[2]//2), int(bbox[1]+bbox[3]//2)), (0, 0, 255), 5)
            flag = False
            tmp = frame
            tmp_bboxs = bboxs
        else:
            flag = True
            annotated_frame = frame.copy()
            for bbox in a:
                cv2.rectangle(annotated_frame, (int(bbox[0]-bbox[2]//2), int(bbox[1]-bbox[3]//2)), (int(bbox[0]+bbox[2]//2), int(bbox[1]+bbox[3]//2)), (0, 255, 0), 3)
            for bbox in bboxs:
                cv2.rectangle(annotated_frame, (int(bbox[0]-bbox[2]//2), int(bbox[1]-bbox[3]//2)), (int(bbox[0]+bbox[2]//2), int(bbox[1]+bbox[3]//2)), (0, 0, 255), 5)
        #annotated_frame = results[0].plot()

        # 绘制追踪路径
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y中心点
            if len(track) > 30:  # 在90帧中保留90个追踪点
                track.pop(0)

            # 绘制追踪线
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=6)

        # 展示带注释的帧
        #cv2.imwrite("YOLOv8 Tracking.jpg", annotated_frame)
        out.write(annotated_frame)

        # 如果按下'q'则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则退出循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
out.release()  #资源释放
cv2.destroyAllWindows()