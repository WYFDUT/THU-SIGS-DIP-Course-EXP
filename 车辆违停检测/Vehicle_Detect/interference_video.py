import cv2
from PIL import Image
from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# 加载YOLOv8模型
model = YOLO('/root/wyf/detection/runs/detect/train4/weights/best.pt')

# 打开视频文件
video_path = "/root/wyf/Violation_Detect/Vehicle_Detect/1_cut.mp4"
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
out = cv2.VideoWriter('Vehicle_detect.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame.shape[1], frame.shape[0]))  # 保存视频

# 遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在该帧上运行YOLOv8推理
        results = model.predict(frame, classes=[3,4,5,8])

        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        # 显示带注释的帧
        out.write(annotated_frame)
        #cv2.imwrite("YOLOv8推理", annotated_frame)

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