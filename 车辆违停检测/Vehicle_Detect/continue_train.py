from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 加载模型
model = YOLO('/root/wyf/detection/runs/detect/train4/weights/last.pt')  # 加载部分训练的模型

# 恢复训练
results = model.train(resume=True)