import torch

from .video import ffmpeg_try_process_video, get_video_info_old, get_video_info


def colmap_poses_to_intrinsics_extrinsics(data):
    # Format description at github.com/Fyusion/LLFF?tab=readme-ov-file#using-your-own-poses-without-running-colmap
    mat, close, far = data[:, :-2].reshape((-1, 3, 5)), data[:, -2], data[:, -1]
    T, mat2 = mat[:, :, :-1], mat[:, :, -1:]
    h, w, f = mat2[:, 0, 0], mat2[:, 1, 0], mat2[:, 2, 0]

    # Since we only have width, height and focal point, the intrinsics matrix will give imprecise results
    K = torch.zeros((T.shape[0], 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 2, 2], K[:, 0, 2], K[:, 1, 2] = f, f, 1, w / 2, h / 2

    return K, T, (h, w)
