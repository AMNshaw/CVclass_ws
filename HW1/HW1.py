import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

image_row = 120
image_col = 120

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
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (N.shape[0], N.shape[1], 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
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
    return N

def surface_reconstruction(N):
    S = image_row*image_col
    Mx = np.zeros((S, S), dtype=np.int8)
    My = np.zeros((S, S), dtype=np.int8)
    V = np.zeros((2*S, 1), dtype = np.float32)

    nx, ny, nz = N.reshape((S, 3)).T

    n = 0
    for i in range (S):
        if nz[i] != 0:
            V[n] = -nx[i]/nz[i]
            V[n+S] = -ny[i]/nz[i]
        n = n+1

    for i in range(image_row):
        for j in range(image_col-1):
            k = i*image_col+j
            Mx[k, k] = -1
            Mx[k, k+1] = 1

    for i in range(image_row-1):
        for j in range(image_col):
            k = i*image_col+j
            My[k, k] = -1
            My[k, k+image_col] = 1

    M = np.vstack((Mx, My))
    M = coo_matrix(M).tocsr()
    z = lsqr(M, V, damp=0.1)[0]

    return z

if __name__ == '__main__':

    [I, L] = readImg_n_lightVec("test/star/")
    N = normal_estimation(I, L)
    normal_visualization(N)
    z = surface_reconstruction(N)
    depth_visualization(z)
    '''

    depth_visualization()
    save_ply(Z,filepath)
    show_ply(filepath)
    '''
    plt.show()