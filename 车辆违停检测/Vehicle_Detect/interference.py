from PIL import Image
from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cv2
import numpy as np
from matplotlib import pyplot as plt
import predict_mbv2 as Classify
import bcp_dcp as dcp


image_path = '/root/wyf/Violation_Detect/Vehicle_Detect/test15.png'
condition = Classify.judge_day_night(image_path)
if condition == "Night":
    print("Night")
    img = cv2.imread(image_path)
    img = dcp.enhance(img)
else:
    print("Day")
    img = cv2.imread(image_path)
    #img = dcp.enhance(img)


# 加载预训练的YOLOv8n模型
model = YOLO('/root/wyf/detection/runs/detect/train4/weights/best.pt')

# 在'bus.jpg'上运行推理
results = model.predict(img, classes=[3,4,5,8])  # 结果列表

# 展示结果
for r in results:
    #print(r.boxes)
    im_array = r.plot()  # 绘制包含预测结果的BGR numpy数组
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
    im.show()  # 显示图像
    im.save('results.jpg')  # 保存图像