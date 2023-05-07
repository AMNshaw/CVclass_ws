import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import random
import math
import sys
import os

# pip install tdqm
# pip install ipywidgets

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ", img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def im_show():
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def SIFT_detection(img):
    SIFT_Detector = cv2.SIFT_create()
    kp, des = SIFT_Detector.detectAndCompute(img, None)
    
    return kp, des
    
def featureMatching(kp0, kp1, des0, des1, threshold = 0.75):
    matches_1to2 = []
    matches_2to1 = []
    best_match = []

    for i in range(len(des0)):
        distances = [euclidean_distance(des0[i], des1[j]) for j in range(len(des1))]
        sorted_indices = np.argsort(distances)
        first_nearest = sorted_indices[0]
        second_nearest = sorted_indices[1]

        if distances[first_nearest] < threshold * distances[second_nearest]:
            matches_1to2.append(i)
            matches_2to1.append(first_nearest)
            best_match.append(list(kp0[i].pt + kp1[first_nearest].pt))


    best_match = np.asarray(best_match)

    return best_match

def Homography(matches):
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

def matches_errorCal(points, H):
    num_points = len(points)
    p1_a = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    p2_a = points[:, 2:4]
    p2_prime = np.zeros((num_points, 2))

    for i in range(num_points):
        p2_prime_raw = H @ p1_a[i]
        p2_prime[i] = (p2_prime_raw/p2_prime_raw[2])[0:2] # scale p2_prime, last column must be 1 
    
    # Compute error
    matches_error = np.linalg.norm(p2_a - p2_prime , axis=1) ** 2

    return matches_error

def RANSAC(matches, threshold = 0.3, iters = 3000):
    most_inliers = 0
    
    for i in range(iters):
        # STEP1, randomly select 4 data points
        points = Randompoints(matches)
        # STEP2, find Homography matrix
        H = Homography(points)
        
        # Implement rank check
        if np.linalg.matrix_rank(H) < 3:
            continue
        
        # Calculate error between p2, p2'
        matches_error = matches_errorCal(matches, H)

        # Get H that fit the most
        index = np.where(matches_error < threshold)[0]
        S = matches[index]
        num_inliers = len(S)
        if num_inliers > most_inliers:
            largest_S = S
            most_inliers = num_inliers
            best_H = H
    
    return largest_S, best_H

def StitchImage(img1, img2, H):
    print("stiching image ...")
    
    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(img1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv2.normalize(img2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    height_r, width_r, channel_r = right.shape
    h = np.minimum(height_l, height_r)
    w = np.minimum(width_l, width_r)
    
    corners = [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]
    corners_new = [H@corner for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                opt = [pixel_l, pixel_r]
                warped_l[i, j, :] = (pixel_l + pixel_r)/2#opt[np.argmax([np.linalg.norm(pixel_l), np.linalg.norm(pixel_r)])]
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    
    return stitch_image

if __name__ == '__main__':
    
    baseline_files = []
    bonus_files = []
    for i in range(0, 6):
        baseline_files.append('m' + str(i+1) + '.jpg')

    for i in range(0, 4):
        bonus_files.append('m' + str(i+1) + '.jpg')
    
    folder = 'bonus/'
    if folder == 'baseline/':
        length = len(baseline_files)
    elif folder == 'bonus/':
        length = len(bonus_files)

    img_res = []
    img_res_gray = []

    for i in range(length):
        if i == 0:
            img0, img_gray0 = read_img(folder + baseline_files[i])

        else:
            img0, img_gray0 = img_res, img_res_gray
        if i < length-1:
            img1, img_gray1 = read_img(folder + baseline_files[i+1])

        kp0, des0 = SIFT_detection(img_gray0)
        kp1, des1 = SIFT_detection(img_gray1)
        
        matches = featureMatching(kp0, kp1, des0, des1, threshold=0.5)
        
        S, H = RANSAC(matches, 0.5, 3000)
        
        img_res = StitchImage(img0, img1, H).astype(np.float32)
        img_res_gray = (cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)*255).astype(np.uint8)
    
        cv2.imwrite('./result/' + folder + 'm'+str(i)+'.jpg', (img_res*255).astype(np.uint8))
        cv2.imwrite('./result/' + folder + 'm_gray'+str(i)+'.jpg', img_res_gray)

        
    creat_im_window("res", img_res)    
    im_show()


    