import cv2

def mask_iou(mask1, mask2):
    area1 = mask1.sum()
    area2 = mask2.sum()
    inter = ((mask1+mask2)==2.0).sum()
    mask_iou = inter / (area1+area2-inter)
    return mask_iou


mask1 = cv2.imread("C:\\Users\\WYF\\Desktop\\geoseg\\GeoSeg\\mask\\3_1.png")/255

mask2 = cv2.imread("C:\\Users\\WYF\\Desktop\\geoseg\\GeoSeg\\mask\\3_2.png")/255
mask1 = cv2.resize(mask1, (mask2.shape[1], mask2.shape[0]))
print(mask_iou(mask1, mask2))
