import os
import cv2
import numpy as np
import harris_corner as HC


def imgShow(img, name='test'):
    cv2.imshow(name, np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sift_method(ori_img, gray):
    img = ori_img.copy()
    sift = cv2.SIFT_create()
    # Find key points
    kp = sift.detect(gray, None)
    # Draw points
    img = cv2.drawKeypoints(img, kp, gray, color=None, flags=None)  
    # Compute vector
    kp, des = sift.compute(gray, kp)   
    return img, kp, des


if __name__ == "__main__":
    img_path = "C:\\Users\\WYF\\Desktop\\DIP\\features_detection\\pic5.png"
    img_path2 = "C:\\Users\\WYF\\Desktop\\DIP\\features_detection\\test1.png"

    ori_img, ori_img2 = cv2.imread(img_path), cv2.imread(img_path2)
    img, img2 = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ori_img2, cv2.COLOR_BGR2GRAY)

    # Use user func
    corner = HC.harrisCorner(ori_img, img)
    imgShow(corner)

    # Use opencv func 
    dst = cv2.cornerHarris(img, 5, 3, 0.04)
    ori_img[np.where(dst>0.01*dst.max())[0], np.where(dst>0.01*dst.max())[1], 2] = 255
    imgShow(ori_img)

    # SIFT
    sift_img1, kp1, des1 = sift_method(ori_img, img)
    sift_img2, kp2, des2 = sift_method(ori_img2, img2)
    bf = cv2.BFMatcher(crossCheck=True)
    
    # 1 to 1 Compare
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(ori_img, kp1, ori_img2, kp2, matches[:-1], None, flags=2)
    imgShow(img3)

    # K pair best matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img3 = cv2.drawMatchesKnn(ori_img, kp1, ori_img2, kp2, good[:-1], None, flags=2)
    imgShow(img3)




