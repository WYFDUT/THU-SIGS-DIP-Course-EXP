import cv2
from PIL import Image
from ultralytics import YOLO
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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





# 加载YOLOv8模型
model = YOLO('/root/wyf/detection/runs/detect/train4/weights/best.pt')

# 打开视频文件
video_path = "/root/wyf/Violation_Detect/Vehicle_Detect/4_cut.mp4"
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
tmp = np.zeros_like(frame)
out = cv2.VideoWriter('Violation_detect2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame.shape[1], frame.shape[0]))  # 保存视频
flag = True
# 遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在该帧上运行YOLOv8推理
        results = model.predict(frame, classes=[3,4,5,8])

        # 在帧上可视化结果
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
        
        # 显示带注释的帧
        out.write(annotated_frame)
        #cv2.imwrite("YOLOv8推理.jpg", annotated_frame)

        # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # 如果视频结束则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
out.release()  #资源释放
cv2.destroyAllWindows()