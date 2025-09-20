import open3d as o3d
import torch

def _get_edges(v):
    return [v[0, :, 0, 0], v[0, :, -1, 0], v[0, :, 0, -1], v[0, :, -1, -1]]

def get_camera_geometry(o, d, d_multiplier=30):
    # Also using point cloud bc for some reason colors in lines are not working
    lset = o3d.geometry.LineSet()
    pcd = o3d.geometry.PointCloud()

    points = torch.stack([o[0, :, 0, 0]] + _get_edges(o + d * d_multiplier), dim=-2).numpy()
    lines = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]).numpy()
    colors = torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]] + [[0, 0, 0]] * 4).numpy()
    colors2 = torch.tensor([[1., 0, 0], [0., 0, 0], [0., 1, 0], [0., 0, 1], [0., 1, 1]]).numpy()

    lset.points = o3d.utility.Vector3dVector(points)
    lset.lines = o3d.utility.Vector2iVector(lines)
    lset.colors = o3d.utility.Vector3dVector(colors)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors2)
    
    return [lset, pcd]
