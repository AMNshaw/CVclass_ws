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

def img_read_recursive(folder):
    directory = 'baseline'
    img_list = []
    img_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img_files.append(os.path.join(directory, filename))

    for img_file in img_files:
        img, img_gray = read_img(img_file)
        img_list.append(img_gray)

    return img_list

def SIFT_detection(img_list):
    SIFT_detector = cv2.SIFT_create()

    des_list = []
    kp_list = []
    for img in img_list:
        kp, des = SIFT_detector.detectAndCompute(img, None)
        des_list.append(des)
        kp_list.append(kp)

    return des_list, kp_list

def featureMatching(img1, img2, des1, des2, kp1, kp2):
    matches_1to2 = []
    matches_2to1 = []
    best_match_kp1 = []
    best_match_kp2 = []

    for i in range(len(des1)):
        distances = [euclidean_distance(des1[i], des2[j]) for j in range(len(des2))]
        sorted_indices = np.argsort(distances)
        best_match_index = sorted_indices[0]
        second_best_match_index = sorted_indices[1]
        if distances[best_match_index] < 0.75 * distances[second_best_match_index]:
            matches_1to2.append(i)
            matches_2to1.append(best_match_index)
            best_match_kp1.append(kp1[i].pt)
            best_match_kp2.append(kp2[best_match_index].pt)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, 
                      [(matches_1to2[i], matches_2to1[i]) for i in range(len(matches_1to2))], None)

    return matches_1to2, matches_2to1

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

    img_list = img_read_recursive("baseline/")
    des_list, kp_list = SIFT_detection(img_list)

    matches_1to2, matches_2to1 = featureMatching(img_list[0], img_list[1], des_list[0], des_list[1], kp_list[0], kp_list[1])

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