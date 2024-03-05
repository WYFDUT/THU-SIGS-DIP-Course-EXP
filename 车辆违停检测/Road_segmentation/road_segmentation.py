import os 
import cv2
import argparse
import numpy as np
import ttach as tta
from pathlib import Path
import torch 
import torch.nn as nn
import geoseg.models.UNetFormer_lsk as Net
import albumentations as albu

import predict_mbv2 as Classify
import night_enhancement as night_en

from torch.utils.data import Dataset, DataLoader

from train_supervision import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    #arg("-i", "--image_path", type=str, default='data/uavid/uavid_test', help="Path to huge image")
    #arg("-c", "--config_path", type=Path, default='/root/wyf/air/GeoSeg/config/uavid/unetformer_lsk_s.py', required=False, help="Path to config")
    arg("-o", "--output_path", type=Path, help="Path to save resulting masks.", required=False)
    arg("-t", "--tta", help="Test time augmentation.", default="lr", choices=[None, "d4", "lr"])
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=512)
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=1024)
    arg("-b", "--batch-size", help="batch size", type=int, default=1)
    arg("-d", "--dataset", help="dataset", default="uavid", choices=["uavid"])
    return parser.parse_args()

def make_dataset_for_one_huge_image(img, patch_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)

    output_height, output_width = image_pad.shape[0], image_pad.shape[1]

    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x:x+patch_size[0], y:y+patch_size[1]]
            tile_list.append(image_tile)
    
    dataset = InferenceDataset(tile_list=tile_list)
    return dataset, width_pad, height_pad, output_width, output_height, image_pad, img.shape

def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(min_height=h, min_width=w, border_mode=0,
                           position='bottom_right', value=[0, 0, 0])(image=image)
    img_pad = pad['image']
    return img_pad, height_pad, width_pad

class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, transform=albu.Normalize()):
        self.tile_list = tile_list
        self.transform = transform

    def __getitem__(self, index):
        img = self.tile_list[index]
        img_id = index
        aug = self.transform(image=img)
        #breakpoint()
        img = aug['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img)
        return results

    def __len__(self):
        return len(self.tile_list)



class RoadSegmentation():
    def __init__(self, img) -> None:

        self.args = get_args()
        self.patch_size = (self.args.patch_height, self.args.patch_width)
        self.transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        
        self.net = Net.UNetFormer_lsk_s()
        self.net.load_state_dict(torch.load("C:\\Users\\WYF\\Desktop\\geoseg\\GeoSeg\\geoseg.pth"))

        self.net.cuda()
        self.net.eval()

        self.net = tta.SegmentationTTAWrapper(self.net, self.transforms)

        self.dataset, self.width_pad, self.height_pad, self.output_width, self.output_height, self.img_pad, self.img_shape = \
                make_dataset_for_one_huge_image(img, self.patch_size)
        
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.args.batch_size,
                                drop_last=False, shuffle=False)

    @staticmethod
    def uavid2rgb(mask):
        h, w = mask.shape[0], mask.shape[1]
        mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        mask_convert = mask[np.newaxis, :, :]
        mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]   #[128, 0, 0]
        mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]       #[128, 64, 128]
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        return mask_rgb

    def maskGenerate(self):
        output_mask = np.zeros(shape=(self.output_height, self.output_width), dtype=np.uint8)
        output_tiles = []
        k = 0
        with torch.no_grad():
            for input in (self.dataloader):
            # raw_prediction NxCxHxW
            #print(input['img'].shape)
                raw_predictions = self.net(input['img'].cuda())
                raw_predictions = nn.Softmax(dim=1)(raw_predictions)
                predictions = raw_predictions.argmax(dim=1)
                image_ids = input['img_id']

                for i in range(predictions.shape[0]):
                    raw_mask = predictions[i].cpu().numpy()
                    #print(predictions[i].shape)
                    mask = raw_mask
                    output_tiles.append((mask, image_ids[i].cpu().numpy()))

        for m in range(0, self.output_height, self.patch_size[0]):
            for n in range(0, self.output_width, self.patch_size[1]):
                output_mask[m:m + self.patch_size[0], n:n + self.patch_size[1]] = output_tiles[k][0]
                k = k + 1

        output_mask = output_mask[-self.img_shape[0]:, -self.img_shape[1]:]
        output_mask = self.uavid2rgb(output_mask)
        return output_mask
    
    def post_process_mask(self, mask):
        # Perform morphological operations to close small gaps and remove noise
        kernel = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        return opening


import time
start = time.time()

torch.manual_seed(42)
torch.cuda.manual_seed(42)
img_path = "C:\\Users\\WYF\\Desktop\\geoseg\\DIPA_road\\DIPA_road\\test\\images\\frame_163.jpg"
save_path = "C:\\Users\\WYF\\Desktop\\geoseg\\results"
condition = Classify.judge_day_night(img_path)

if condition == "Night":
    print("Night")
    img = cv2.imread(img_path)
    img_en = night_en.night_image_enhance(img_path)
    # Do not resize the input, such operation will affect model performance 
    # img = cv2.resize(img,(640,640))
    # print(img.shape)

    r = RoadSegmentation(img_en)
    map = r.maskGenerate()
    map_2,_ = night_en.segment_bright_parts(None, img_en)
    map = cv2.bitwise_or(map_2, map)
    map = r.post_process_mask(map)
    img_and = cv2.bitwise_and(img, map)

    cv2.imwrite(os.path.join(save_path, "frame_163.jpg"), map)
    cv2.imwrite(os.path.join(save_path, "frame_163_and.jpg"), img_and)

else:
    print("Day")
    img = cv2.imread(img_path)
    #img_en = img
    # Do not resize the input, such operation will affect model performance 
    # img = cv2.resize(img,(640,640))
    # print(img.shape)

    r = RoadSegmentation(img)
    map = r.maskGenerate()
    #img_en = night_en.night_image_enhance(img_path)
    #map_2,_ = night_en.segment_bright_parts(None, img_en)
    #map = cv2.bitwise_or(map_2, map)
    map = r.post_process_mask(map)
    img_and = cv2.bitwise_and(img, map)

    cv2.imwrite(os.path.join(save_path, "frame_163.jpg"), map)
    cv2.imwrite(os.path.join(save_path, "frame_163_and.jpg"), img_and)

end=time.time()
print(end-start)











