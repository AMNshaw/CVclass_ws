import cv2
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import lsqr
import scipy
from scipy.integrate import dblquad
from math import acos
import random

image_row = 120
image_col = 120
name = "venus"

def imgRead_recursive(folder):
    global image_row
    global image_col
    n = 1
    file = "pic"
    extension = ".bmp"
    I = []

    img_gray = cv2.imread(folder + file + str(n) + extension, cv2.IMREAD_GRAYSCALE)
    image_row, image_col = img_gray.shape

    while cv2.imread(folder + file + str(n) + extension, cv2.IMREAD_GRAYSCALE) is not None:
        img_gray = cv2.imread(folder + file + str(n) + extension, cv2.IMREAD_GRAYSCALE)
        I.append(np.array(img_gray))
        n = n + 1

    I = np.array(I)
        
    return I

def lightVec_read_recursive(folder):
    file_name = "LightSource.txt"
    path = folder + file_name
    intensities = []

    with open(path, "r") as file:
        lines = file.readlines()

    L = []
    for line in lines:
        value = line.split(':')[1].strip()[1:-1]
        L.append(tuple(map(int, value.split(','))))

    for i in range(len(L)):
        vector = np.array(L[i])
        norm = np.sqrt(np.sum(vector**2))
        L[i] = L[i]/norm

    L = np.array(L)

    return L

def readImg_n_lightVec(folder):
    I = imgRead_recursive(folder)
    L = lightVec_read_recursive(folder)
    return I, L

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.imsave("result/"+name+"_mask.png", mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.imsave("result/"+name+"_normal.png", N_map)
    plt.title('Normal map')


# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.imsave("result/"+name+"_depth.png", D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.int16)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

def normal_estimation(I, L):
    
    L_psinv = np.linalg.inv(L.T@L)@L.T
    I_T = np.transpose(I, (1, 0, 2))
    Kd_N = np.transpose(L_psinv@I_T, (0, 2, 1))
    N = np.zeros((Kd_N.shape[0], Kd_N.shape[1], Kd_N.shape[2]))

    for i in range(Kd_N.shape[0]):
        for j in range(Kd_N.shape[1]):
            if np.linalg.norm(Kd_N[i][j]) > 0:
                N[i][j] = Kd_N[i][j]/np.linalg.norm(Kd_N[i][j])

    N = N.reshape((image_row*image_col, 3))
    return N

def CreateMask(normal_map):
    map_1D = np.reshape(np.sum(normal_map, axis = 1), ((image_row,image_col))).copy()
    mask = np.where(map_1D != 0, 1, 0)
    return mask

def depth_map_construction(normal_map, mask):
    normal_map = np.copy(np.reshape(normal_map, (image_row, image_col, 3)))
    [Y, X] = np.where(mask != 0)
    coord = np.dstack((Y,X))[0]
    S = len(coord)
    M = np.zeros((2*S,S))
    V = np.zeros((2*S,1))

    idx_map = np.zeros((mask.shape), dtype=np.int64)
    for idx in range(S):
        idx_map[coord[idx][0], coord[idx][1]] = int(idx)

    for idx in range(S):
        x = coord[idx][1]
        y = coord[idx][0]
        n = normal_map[y,x]

        if mask[y,x+1] > 0 and mask[y-1,x] > 0:
            M[idx, idx] = -1
            tmp = idx_map[y,x+1]
            M[idx, tmp] = 1
            V[idx] = -n[0]/n[2]

            M[idx+S, idx] = -1
            tmp = idx_map[y-1,x]
            M[idx+S, tmp] = 1
            V[idx+S] = -n[1]/n[2]

        # (x+1, y) is not valid
        elif mask[y-1,x] > 0:
            if mask[y, x-1] > 0:
                M[idx, idx] = -1
                tmp = idx_map[y,x-1]
                M[idx, tmp] = 1
                V[idx] = n[0]/n[2]

            M[idx+S, idx] = -1
            tmp = idx_map[y-1,x]
            M[idx+S, tmp] = 1
            V[idx+S] = -n[1]/n[2]

        # (x, y+1) is not valid
        elif mask[y, x+1] > 0:
            if mask[y+1,x] > 0:
                M[idx+S, idx] = -1
                tmp = idx_map[y+1,x]
                M[idx+S, tmp] = 1
                V[idx+S] = n[1]/n[2]

            M[idx, idx] = -1
            tmp = idx_map[y,x+1]
            M[idx, tmp] = 1
            V[idx] = -n[0]/n[2]

        # both is not valid
        else:
            if mask[y+1,x] > 0:
                M[idx+S, idx] = -1
                tmp = idx_map[y+1,x]
                M[idx+S, tmp] = 1
                V[idx+S] = n[1]/n[2]

            if mask[y, x-1] > 0:
                M[idx, idx] = -1
                tmp = idx_map[y,x-1]
                M[idx, tmp] = 1
                V[idx] = n[0]/n[2]

    M = csr_matrix(M)
    Z = scipy.sparse.linalg.lsqr(M,V)[0]

    D = []
    for d in Z:
        if d > 0:
            D.append(d)

    z = np.zeros((image_row,image_col))

    idx = 0
    for p in coord:
        z[p[0], p[1]] = Z[idx]
        idx += 1

    return z

if __name__ == '__main__':

    result_path = 'result/'+name+'.ply'

    [I, L] = readImg_n_lightVec("test/"+name+"/")

    N = normal_estimation(I, L)
    normal_visualization(N)
    M = CreateMask(N)
    mask_visualization(M)
    z = depth_map_construction(N, M);
    depth_visualization(z)

    save_ply(z,result_path)
    show_ply(result_path)

    plt.show()