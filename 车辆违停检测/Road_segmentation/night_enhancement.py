import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import threadpoolctl

# Disable threadpoolctl temporarily
threadpoolctl.threadpool_limits(1)


def clahe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab_image[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahed_l_channel = clahe.apply(l_channel)
    lab_image[:, :, 0] = clahed_l_channel
    clahed_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    return clahed_image


def night_image_enhance(img_path):

    image = cv2.imread(img_path)
    # Apply preprocessing steps
    image = clahe(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #img, mask = segment_bright_parts(image)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #cv2.imwrite(f"images/{img_name}_enhanced.jpg", cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.figure(figsize=(12,10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    return image


def segment_bright_parts(img_path, img, threshold=150):
    # Convert the image to grayscale
    if img_path == None:
        image = img
    else:
        image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment bright parts
    _, binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Apply the binary mask to the original image
    segmented_image = cv2.bitwise_and(image, image, mask=binary_mask)
    #plt.figure(figsize=(12,10))
    #plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    #plt.show()

    road_img = detect_lanes(segmented_image)
    plt.figure(figsize=(12,10))
    plt.imshow(cv2.cvtColor(road_img, cv2.COLOR_BGR2RGB))
    plt.show()
    return road_img, segmented_image


def detect_lanes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=600, maxLineGap=40)

    # Draw detected lines on a blank image
    lanes_image = np.zeros_like(image)

    try:
        if lines == None:
            return np.zeros_like(image).astype("uint8")
    except:   
        # Create a blank mask
        mask = np.zeros_like(image)

        # Create a list of points for each detected lane segment
        lane_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lane_points.extend([(x1, y1), (x2, y2)])

        # Convert the list to a NumPy array
        lane_points = np.array(lane_points, dtype=np.int32)
        
        # Reshape the array to have dimensions (number_of_points, 1, 2)
        lane_points = lane_points.reshape((-1, 2))

        pts_add = []
        min_x, max_x, min_y, max_y = np.argmin(lane_points[:,0]), np.argmax(lane_points[:,0]), np.argmin(lane_points[:,1]), np.argmax(lane_points[:,1])
        pts_add.extend([lane_points[min_x], lane_points[max_x], lane_points[min_y], lane_points[max_y]])
        pts_add = np.array(pts_add, dtype=np.int32).reshape((-1, 2))

        # Fill the area between the lane lines with white color (255)
        cv2.fillPoly(mask, [lane_points], color=(255, 255, 255))
        #cv2.fillPoly(mask, [pts_add], color=(255, 255, 255))

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        tmp = np.zeros_like(image)
        tmp[:,:,0], tmp[:,:,1], tmp[:,:,2] = mask, mask, mask
        return tmp



if __name__ == "__main__":
    img = night_image_enhance("C:\\Users\\WYF\\Desktop\\geoseg\\DIPA_road\\DIPA_road\\test\\images\\frame_35.jpg")
    segment_bright_parts(img_path=None, img=img)
    #img = cv2.imread("C:\\Users\\WYF\\Desktop\\geoseg\\DIPA_road\\DIPA_road\\test\\images\\frame_35.jpg")
    #day_img_lanes(img)