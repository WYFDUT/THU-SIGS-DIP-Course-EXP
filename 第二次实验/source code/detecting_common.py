import os 
import cv2
import math
import time 
import tqdm
import fnmatch
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pylab as plt
from collections import defaultdict


class DetectCommon():
    def __init__(self) -> None:
        pass

    def cart2polar(self, region_size):
        radius, angle = np.zeros((region_size[0], region_size[1])), np.zeros((region_size[0], region_size[1]))
        center = [math.ceil((region_size[0]/2)), math.ceil((region_size[1]/2))]
        for row in range(region_size[0]):
            for col in range(region_size[1]):
                x = row - center[0]
                y = col - center[1]

                rho = np.log(np.sqrt(x**2 + y**2))
                theta = (np.arctan2(x, y) * 180 / np.pi) + 180

                radius[row, col] = rho
                angle[row, col] = theta
        return radius, angle
    
    def cal_ssd(self, patch, region, alpha, center_patch):
        patch_size = patch.shape
        region_size = region.shape
        SSD_region = np.zeros(region_size[:2])

        for row in range(center_patch[0], region_size[0] - center_patch[0]):
            for col in range(center_patch[1], region_size[1] - center_patch[1]):
                temp = region[row - center_patch[0]:row + center_patch[0] + 1,
                            col - center_patch[1]:col + center_patch[1] + 1, :] - patch[:, :, :]
                SSD_region[row, col] = np.sum(np.sum(np.sum(temp**2)))
                SSD_region[row, col] = np.exp(-alpha * SSD_region[row, col])

        return SSD_region

    def get_bin(self, radius, angle, region_size):
        max_radius = np.max(np.max(radius))  # Maximum radius
        bins = [[None for _ in range(3)] for _ in range(15)]  # Initialize the bins

        for m in range(15):
            theta_low = m * 24
            theta_up = (m + 1) * 24
            for n in range(3):
                rho_low = max_radius * n / 3
                rho_up = max_radius * (n + 1) / 3

                # Loop through the entire region to find positions belonging to the same bin
                temp = []
                num = 0
                for row in range(region_size[0]):
                    for col in range(region_size[1]):
                        if (rho_low <= radius[row, col] <= rho_up) and (theta_low <= angle[row, col] <= theta_up):
                            num += 1
                            temp.append([row, col])

                bins[m][n] = temp
        
        return np.array(bins)
    
    def get_self_sim_vec(self, ssd_region, bin, vec_size):
        self_similarities_vec = np.zeros(vec_size)  # Initialize the descriptor
        num = 0
        for m in range(15):
            for n in range(3):
                temp = bin[m][n]
                max_value = 0

                # Find the maximum value within the same bin
                for loc in temp:
                    row, col = loc
                    max_value = max(max_value, ssd_region[row, col])

                num += 1
                self_similarities_vec[num - 1] = max_value

        return self_similarities_vec
    
    def cal_Self_Similarities(self, src_image, region_size, patch_size, bin):
        # lab_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2Lab)
        lab_image = src_image
        '''
        cv2.imshow("lab_image", lab_image)
        cv2.waitKey()
        cv2.destroyAllWindows()  
        '''
        vec_size = 45
        # Calculate alpha
        alpha = 1 / (85 ** 2)

        # Calculate center offsets
        center_region = ((region_size[0] // 2), (region_size[1] // 2))
        center_patch = ((patch_size[0] // 2), (patch_size[1] // 2))

        # Paddding the image 
        # lab_image = cv2.copyMakeBorder(lab_image, center_region[0], center_region[0], center_region[1], center_region[1], cv2.BORDER_CONSTANT, value=0)
        # Initialize the self-similarity descriptor
        lab_img_h, lab_img_w, lab_img_c  = lab_image.shape
        # self_similarities = np.zeros((lab_img_h-2*center_region[0], lab_img_w-2*center_region[1], vec_size), dtype=np.float32)
        self_similarities = np.zeros((lab_img_h, lab_img_w, vec_size), dtype=np.float32)

        for row in range(center_region[0], lab_img_h - center_region[0]):
            for col in range(center_region[1], lab_img_w - center_region[1]):
                patch = lab_image[(row - center_patch[0]):(row + center_patch[0] + 1), 
                                (col - center_patch[1]):(col + center_patch[1] + 1), :]
                
                region = lab_image[(row - center_region[0]):(row + center_region[0] + 1),
                                (col - center_region[1]):(col + center_region[1] + 1), :]
                
                SSD_region = self.cal_ssd(patch, region, alpha, center_patch)
                vec = self.get_self_sim_vec(SSD_region, bin, vec_size)
                LSSD = cv2.normalize(vec, None, 0, 1, cv2.NORM_MINMAX).flatten()
                # print(LSSD.shape, self_similarities.shape)
                self_similarities[row, col, :] = LSSD
        
        return self_similarities
    
    def draw_result(self, src_img, sig_score_img, region_size, scale):
        ma = np.max(sig_score_img)
        mi = np.min(sig_score_img)

        # Normalize the score image
        norm_sig_score_img = (sig_score_img - mi) / (ma - mi)
        norm_sig_score_img = (norm_sig_score_img * 255).astype(np.uint8)

        # Resize the normalized score image
        norm_sig_score_img = cv2.resize(norm_sig_score_img, None, fx=scale, fy=scale)

        # Find the coordinates of the maximum score
        x, y = np.unravel_index(np.argmax(sig_score_img), sig_score_img.shape)
        # x, y = np.unravel_index(np.argmax(norm_sig_score_img), norm_sig_score_img.shape)

        new_img = src_img.copy()

        new_img[(x - region_size[0] // 2)*scale:(x + region_size[0] // 2)*scale, (y - region_size[1] // 2)*scale, 0] = 255
        new_img[(x - region_size[0] // 2)*scale:(x + region_size[0] // 2)*scale, (y - region_size[1] // 2)*scale, 1] = 0
        new_img[(x - region_size[0] // 2)*scale:(x + region_size[0] // 2)*scale, (y - region_size[1] // 2)*scale, 2] = 0
        new_img[(x - region_size[0] // 2)*scale:(x + region_size[0] // 2)*scale, (y + region_size[1] // 2)*scale, 0] = 255
        new_img[(x - region_size[0] // 2)*scale:(x + region_size[0] // 2)*scale, (y + region_size[1] // 2)*scale, 1] = 0
        new_img[(x - region_size[0] // 2)*scale:(x + region_size[0] // 2)*scale, (y + region_size[1] // 2)*scale, 2] = 0
        new_img[(x - region_size[0] // 2)*scale, (y - region_size[1] // 2)*scale:(y + region_size[1] // 2)*scale, 0] = 255
        new_img[(x - region_size[0] // 2)*scale, (y - region_size[1] // 2)*scale:(y + region_size[1] // 2)*scale, 1] = 0
        new_img[(x - region_size[0] // 2)*scale, (y - region_size[1] // 2)*scale:(y + region_size[1] // 2)*scale, 2] = 0
        new_img[(x + region_size[0] // 2)*scale, (y - region_size[1] // 2)*scale:(y + region_size[1] // 2)*scale, 0] = 255
        new_img[(x + region_size[0] // 2)*scale, (y - region_size[1] // 2)*scale:(y + region_size[1] // 2)*scale, 1] = 0
        new_img[(x + region_size[0] // 2)*scale, (y - region_size[1] // 2)*scale:(y + region_size[1] // 2)*scale, 2] = 0
        
        cv2.imshow("result", new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return norm_sig_score_img, new_img


if __name__ == "__main__":
    region_size = [45, 37]
    patch_size = [5, 5]

    imgs, imgs_name = [], []
    self_similarities = []
    detect_common = DetectCommon()
    width, height = 1, 1
    center_sub = [width // 2, height // 2]

    img_path = "C:\\Users\\WYF\\Desktop\\DIP\\detect_common\\image"
    detect_result_path = "C:\\Users\\WYF\\Desktop\\DIP\\detect_common"

    imgfile_rootpath = [name for name in os.listdir(img_path)]
    for index, img_name in enumerate(tqdm.tqdm(imgfile_rootpath, desc='Cauculate self similarity')):
        if fnmatch.fnmatch(img_name, '*.jpg') or fnmatch.fnmatch(img_name, '*.png'):
            imgs_name.append(img_name)
            img = cv2.imread(os.path.join(img_path, img_name))
            imgs.append(cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA))
        else:
            break
        img = imgs[index]
        # print(img.shape)
        start_time = time.time()
        radius, angle = detect_common.cart2polar(region_size)
        bin = detect_common.get_bin(radius, angle, region_size)

        self_similarity = detect_common.cal_Self_Similarities(img, region_size, patch_size, bin)
        end_time =  time.time()
        print("Running time is {} s".format(end_time-start_time))
        self_similarities.append(self_similarity)
        print(self_similarity.shape, img.shape)

    for m in tqdm.tqdm(range(len(imgs)), desc='Detect common'):
        self_similarities1 = self_similarities[m]
        src_img = imgs[m]
        sig_score_img = np.zeros((self_similarities1.shape[0], self_similarities1.shape[1]))
        for row in range(center_sub[0], src_img.shape[0]-center_sub[0]-1):
            for col in range(center_sub[1], src_img.shape[1]-center_sub[1]-1):
                sub1 = self_similarities1[row - center_sub[0]:row + center_sub[0] + 1,
                                        col - center_sub[1]:col + center_sub[1] + 1, :]
                max_match = list(np.zeros(len(imgs) - 1))
                num_img = 0
                match_score = []

                for n in range(len(imgs)):
                    if n != m:
                        self_similarities2 = self_similarities[n]
                        temp1 = np.tile(sub1, (self_similarities2.shape[0], self_similarities2.shape[1], 1))
                        temp2 = -np.sum((self_similarities2 - temp1)**2, axis=2)
                        max_match[num_img] = np.max(temp2)
                        match_score.append(temp2.flatten())
                        num_img += 1

                temp3 = match_score
                avg_match = [np.mean(temp3[i]) for i in range(len(temp3))]
                std_match = [np.std(temp3[i]) for i in range(len(temp3))]
                sig_score_img[row, col] = sum([((max_match[i] - avg_match[i]) / std_match[i]) for i in range(len(avg_match))])
        print(sig_score_img.shape, src_img.shape)

        norm_sig_score_img, new_img = detect_common.draw_result(cv2.imread(os.path.join(img_path, imgs_name[m])), sig_score_img, [45, 37], 3)
        cv2.imwrite(os.path.join(detect_result_path, "SigScoreImg{}.jpg".format(m)), norm_sig_score_img)
        cv2.imwrite(os.path.join(detect_result_path, "DetectImg{}.jpg".format(m)), new_img)
    
    