# -*- encoding: utf-8 -*-
'''
@File    :   point_cloud.py
@Time    :   2025/10/18 02:41:55
@Author  :   crab 
@Version :   1.0
@Desc    :   包含对rgb空间的点云处理, 加载，保存的相关操作
'''
import torch
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

# ================== 点云处理相关 ==========================
def add_points(coords:np.ndarray, rgb:np.ndarray, num_points=1000000):
    """
    向点云添加随机点, 一定阈值范围内增密
    
    Args:
        coords (np.ndarray): 原始点云的坐标，形状为 (N, 3)
        rgb (np.ndarray): 原始点云的RGB颜色, 形状为 (N, 3)
        num_points (int, optional): 要添加的候选随机点的数量. Defaults to 1000000.
    
    Returns:
        tuple: 包含新点云坐标和RGB颜色的元组, 形状分别为 (N+num_points, 3) 和 (N+num_points, 3)
    """
    pcd_by3d = o3d.geometry.PointCloud()
    pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
    bbox = pcd_by3d.get_axis_aligned_bounding_box()
    kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

    np.random.seed(0)
    points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))

    points_inside = []
    color_inside= []
    for point in points:
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
        if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
            points_inside.append(point)
            color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

    all_candicate_coords = np.array(points_inside)
    all_candicate_rgb = np.array(color_inside)
    all_coords = np.concatenate([all_candicate_coords,coords],axis=0)
    all_rgb = np.concatenate([all_candicate_rgb,rgb],axis=0)
    return all_coords, all_rgb


# ================== 点云数据io相关 ==========================
def load_from_pcd(load_path:str):
    """从基础的点云文件加载数据
    
    Returns:
        tuple: 包含点云坐标和RGB颜色的元组, 形状分别为 (N, 3) 和 (N, 3)
    """
    plydata = PlyData.read(load_path)
    
    # 处理坐标, 移动到图像中心
    vertices = plydata["vertex"]
    coords = np.vstack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).T
    x_mean = np.mean(coords[:, 0])
    y_mean = np.mean(coords[:, 1])
    z_mean = np.mean(coords[:, 2])
    coords -= np.array([x_mean, y_mean, z_mean])

    # 处理颜色矩阵
    if vertices.__contains__("red"):
        rgb = (
            np.vstack(
                [vertices["red"], vertices["green"], vertices["blue"]]
            ).T
            / 255.0
        )
    else:
        shs = np.random.random((coords.shape[0], 3)) / 255.0
        rgb = _SH2RGB(shs)
    return coords, rgb

def load_from_shape(prompt:str, save_gif_path:str = None):
    """使用shap-e生成初始的点云
    
    Args:
        prompt (str): 输入的文本提示
        save_gif_path (str, optional): 保存生成的gif文件的路径(包含文件名). Defaults to None.

    Returns:
        tuple: 包含点云坐标和RGB颜色的元组, 形状分别为 (N, 3) 和 (N, 3)
    """
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
    from shap_e.util.notebooks import decode_latent_mesh
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
    diffusion = diffusion_from_config_shape(load_config('diffusion'))

    batch_size = 1
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    if save_gif_path is not None:
        render_mode = 'nerf' # you can change this to 'stf'
        size = 256 # this is the size of the renders; higher values take longer to render.
        cameras = create_pan_cameras(size, device)
        shapeimages = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)
        from .io import save_gif_to_file
        save_gif_to_file(shapeimages, save_gif_path)
    pc = decode_latent_mesh(xm, latents[0]).tri_mesh()

    skip = 1
    coords = pc.verts
    rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

    coords = coords[::skip]
    rgb = rgb[::skip]
    return coords, rgb

def load_from_smpl(load_path:str, num_pts = 50000):
    """从SMPL模型加载点云数据
    
    Args:
        load_path (str): SMPL模型文件路径
        num_pts (int, optional): 采样的点云数量. Defaults to 50000.

    Returns:
        tuple: 包含点云坐标和RGB颜色的元组, 形状分别为 (N, 3) 和 (N, 3)
    """
    mesh = o3d.io.read_triangle_mesh(load_path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_pts)
    coords = np.array(point_cloud.points)
    shs = np.random.random((num_pts, 3)) / 255.0
    rgb = _SH2RGB(shs)
    adjusment = np.zeros_like(coords)
    adjusment[:,0] = coords[:,2]
    adjusment[:,1] = coords[:,0]
    adjusment[:,2] = coords[:,1]
    current_center = np.mean(adjusment, axis=0)
    center_offset = -current_center
    adjusment += center_offset
    return adjusment, rgb

def load_from_3dgs(load_path:str):
    """从3DGS文件加载点云数据
    
    Args:
        load_path (str): 3DGS文件路径

    Returns:
        tuple: 包含点云坐标和RGB颜色的元组, 形状分别为 (N, 3) 和 (N, 3)
    """
    plydata = PlyData.read(load_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = _SH2RGB(features_dc[:,:,0])
    return xyz, color


def save_ply(path, xyz, rgb):
    """保存点云到ply文件"""
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# ================== 文件内部复用的一些小函数 ==========================
def _SH2RGB(sh:np.ndarray):
    """将球谐函数转换为RGB颜色

    Args:
        sh (np.ndarray): 球谐函数，形状为 (num_pts, 3)

    Returns:
        np.ndarray: RGB颜色，形状为 (num_pts, 3)
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5