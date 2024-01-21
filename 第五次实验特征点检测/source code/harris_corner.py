import cv2
import numpy as np


def xdiff(img, ksize=3, scale=1.0, borderType=cv2.BORDER_DEFAULT):
    #x_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) * 1/9
    #fx = cv2.filter2D(img, None, x_kernel)
    fx=cv2.Sobel(img,-1,dx=1,dy=0,ksize=ksize,scale=scale,borderType=borderType)
    return fx

def ydiff(img, ksize=3, scale=1.0, borderType=cv2.BORDER_DEFAULT):
    #y_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) * 1/9
    #fy = cv2.filter2D(img, None, y_kernel)
    fy=cv2.Sobel(img,-1,dx=0,dy=1,ksize=ksize,scale=scale,borderType=borderType)
    return fy


def harrisCorner(img_ori, img_src, block_size=5, aperture_size=3, k=0.04, borderType=cv2.BORDER_DEFAULT, threshold=0.01):
	new_img = img_ori.copy()
	R_arr = np.zeros(img_src.shape,dtype=np.float32)
	img = img_src.astype(np.float32)
    # Scale for gradient cauculate
	scale = 1.0/((aperture_size-1)*2*block_size*255)
	# Sobel cauculate gradient
	Ix = xdiff(img, aperture_size, scale, borderType)
	Iy = ydiff(img, aperture_size, scale, borderType)
	Ixx, Iyy, Ixy = Ix**2, Iy**2, Ix*Iy
	# Sum in window using mean filter
	f_xx = cv2.boxFilter(Ixx,ddepth=-1,ksize=(block_size,block_size) ,anchor =(-1,-1),normalize=False,borderType=borderType)
	f_yy = cv2.boxFilter(Iyy,ddepth=-1,ksize=(block_size,block_size),anchor =(-1,-1),normalize=False,borderType=borderType)
	f_xy = cv2.boxFilter(Ixy, ddepth=-1,ksize=(block_size,block_size),anchor =(-1,-1),normalize=False,borderType=borderType)
	radius = int((block_size - 1) / 2)  
	N_pre = radius
	N_post = block_size - N_pre - 1
	row_s, col_s = N_pre, N_pre
	row_e, col_e = img.shape[0] - N_post, img.shape[1] - N_post

	for r in range(row_s, row_e):
		for c in range(col_s, col_e):
			sum_xx = f_xx[r,c]
			sum_yy = f_yy[r, c]
			sum_xy = f_xy[r, c]
			M = np.array([[sum_xx, sum_xy], [sum_xy, sum_yy]])
            
			R_arr[r,c] = np.linalg.det(M) - (k * (np.trace(M))**2 )
	new_img[np.where(R_arr>threshold*R_arr.max())[0], np.where(R_arr>threshold*R_arr.max())[1], 2] = 255
	return new_img
    

