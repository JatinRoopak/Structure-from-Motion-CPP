import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

DATASET = "Barn" #either Truck, Barn, Meeting_Room
BASE_DIR = f"../outputs/{DATASET}/"

print("Loading Data from C++ Pipeline...")
cameras = pd.read_csv(BASE_DIR + 'ba_cameras.csv').values
points_3d = pd.read_csv(BASE_DIR + 'ba_points.csv').values
observations = pd.read_csv(BASE_DIR + 'ba_obs.csv').values

with open(BASE_DIR + 'intrinsics.txt', 'r') as f:
    fx, fy, cx, cy = map(float, f.read().split())

cam_indices = observations[:, 0].astype(int)
pt_indices = observations[:, 1].astype(int)
points_2d = observations[:, 2:4]

n_cameras = cameras.shape[0]
n_points = points_3d.shape[0]

def rotate(points, rot_vecs):
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params, fx, fy, cx, cy):
    points_cam = rotate(points, camera_params[:, 1:4]) + camera_params[:, 4:7]
    points_proj = np.empty((points.shape[0], 2))
    points_proj[:, 0] = points_cam[:, 0] * fx / points_cam[:, 2] + cx
    points_proj[:, 1] = points_cam[:, 1] * fy / points_cam[:, 2] + cy
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, fx, fy, cx, cy):
    camera_params = params[:n_cameras * 7].reshape((n_cameras, 7))
    points_3d = params[n_cameras * 7:].reshape((n_points, 4))
    points_proj = project(points_3d[point_indices, 1:4], camera_params[camera_indices], fx, fy, cx, cy)
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 7 + n_points * 4
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(1, 7): # rx, ry, rz, tx, ty, tz
        A[2 * i, camera_indices * 7 + s] = 1
        A[2 * i + 1, camera_indices * 7 + s] = 1
    for s in range(1, 4): # x, y, z
        A[2 * i, n_cameras * 7 + point_indices * 4 + s] = 1
        A[2 * i + 1, n_cameras * 7 + point_indices * 4 + s] = 1
    return A

x0 = np.hstack((cameras.ravel(), points_3d.ravel()))
f0 = fun(x0, n_cameras, n_points, cam_indices, pt_indices, points_2d, fx, fy, cx, cy)
print(f"Mean Reprojection Error Before Global BA: {np.mean(np.abs(f0)):.4f} pixels")

A = bundle_adjustment_sparsity(n_cameras, n_points, cam_indices, pt_indices)

print("Running Sparse SciPy Optimizer...")
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', 
                    args=(n_cameras, n_points, cam_indices, pt_indices, points_2d, fx, fy, cx, cy))

print(f"Mean Reprojection Error After Global BA: {np.mean(np.abs(res.fun)):.4f} pixels")

refined_points = res.x[n_cameras * 7:].reshape((n_points, 4))
with open(BASE_DIR + 'ba_cloud_refined.ply', 'w') as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {n_points}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    for pt in refined_points:
        f.write(f"{pt[1]} {pt[2]} {pt[3]}\n")
print(f"saving optimized point cloud to {BASE_DIR}ba_cloud_refined.ply")