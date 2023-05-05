import cv2
import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt
import os


# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def SIFT_detection(img_list):
    SIFT_detector = cv2.SIFT_create()
    kp, des = SIFT_detector.detectAndCompute(img, None)
    return des, kp

def featureMatching(des1, kp1, des2, kp2):
    matches_1to2 = []
    matches_2to1 = []
    best_match = []

    for i in range(len(des1)):
        distances = [euclidean_distance(des1[i], des2[j]) for j in range(len(des2))]
        sorted_indices = np.argsort(distances)
        first_nearest = sorted_indices[0]
        second_nearest = sorted_indices[1]

        if distances[first_nearest] < 0.75 * distances[second_nearest]:
            matches_1to2.append(i)
            matches_2to1.append(first_nearest)
            best_match.append(list(kp1[i].pt + kp2[first_nearest].pt))

    return best_match

def Homography(match_pairs):
    rows = []
    for i in range(matches.shape[0]):
        p1 = np.append(matches[i][0:2], 1)
        p2 = np.append(matches[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] 
    
    return H

def Randompoints(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    
    return np.array(point)

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

if __name__ == '__main__':

    img1, img_gray1 = read_img('baseline/m1');
    img2, img_gray2 = read_img('baseline/m2');
    des1, kp1 = SIFT_detection(img_gray1)
    des2, kp2 = SIFT_detection(img_gray2)

    best_match = featureMatching(des1, kp1, des2, kp2)

    """
    # Draw matches
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)

    plt.imshow(img3),plt.show()
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)

    """