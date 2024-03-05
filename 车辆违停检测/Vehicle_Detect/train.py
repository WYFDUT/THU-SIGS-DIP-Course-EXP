from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='VisDrone.yaml', epochs=100, imgsz=640)

