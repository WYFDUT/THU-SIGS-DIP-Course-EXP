import torch
from torchvision import models
import cv2
import argparse


def load_model():
    # loading imagenet pretrained model from torchvision models
    mbv2 = models.mobilenet_v2(pretrained=True)
    in_features = mbv2.classifier[1].in_features
    # replacing final FC layer of pretrained model with our FC layer having output classes = 2 for day/night
    mbv2.classifier[1] = torch.nn.Linear(in_features, 2)
    # Load trained model onto GPU
    mbv2.load_state_dict(torch.load('/root/wyf/Violation_Detect/Vehicle_Detect/mbv2_best_model.pth', map_location=torch.device('cuda:0')))
    # Setting model to evaluation mode
    mbv2.eval()
    return mbv2

def judge_day_night(img_path):
    #ap = argparse.ArgumentParser()
    #ap.add_argument('-i', '--img', default="C:\\Users\\WYF\\Desktop\\geoseg\\GeoSeg\\test_img\\test_other.jpg", help='Path to image')
    #args = vars(ap.parse_args())

    # imagenet mean and std to normalize data
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    
    # loading model
    model = load_model()

    # reading image
    if type(img_path) == str:
        ori_img = cv2.imread(str(img_path))
    else:
        ori_img = img_path
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    # resizing image to standard size
    img = cv2.resize(img, (500,500))
    # changing order of channels to (channel, height, width) format used by PyTorch
    img = torch.tensor(img).permute(2,0,1)
    # normalizing image
    img = img / 255.0
    img = (img - mean)/std
    img = img.unsqueeze(0)
    out = model(img)
    pred = torch.argmax(out)
    label = 'Day' if pred == 0 else 'Night'
    #print(label)
    return label

if __name__ == '__main__':
    judge_day_night("C:\\Users\\WYF\\Desktop\\geoseg\\GeoSeg\\test_img\\test_other.jpg")