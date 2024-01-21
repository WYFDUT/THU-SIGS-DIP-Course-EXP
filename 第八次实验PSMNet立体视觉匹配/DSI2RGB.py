# coding=utf-8
import cv2
import numpy as np
 
# 读取视差图
disparity_map = cv2.imread('save_images/000132_10.png', cv2.IMREAD_UNCHANGED)
 
# 将视差图转换为深度图
depth_map = np.zeros(disparity_map.shape, dtype=np.float32)
invalid_mask = disparity_map == 0
valid_mask = np.logical_not(invalid_mask)
depth_map[valid_mask] = 1.0 / (disparity_map[valid_mask].astype(np.float32) / 255.0)
 
# 将深度图转换为彩色深度图
depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
depth_map_colored = cv2.applyColorMap(np.uint8(depth_map_normalized * 255), cv2.COLORMAP_JET)
 
# 显示彩色深度图
cv2.imshow('Colored Depth Map', depth_map_colored)
cv2.waitKey(0)