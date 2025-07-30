import importlib

import torch
from torchcodec.decoders import VideoDecoder


def create_bound_function(self, func):
    # Binds an external function to a class instance as if it was defined in the class
    def new_func(*args, **kwargs): return func(self, *args, **kwargs)
    return new_func


def import_object(full_name):
    s = full_name.split('.')
    path, name = '.'.join(s[:-1]), s[-1]
    return importlib.import_module(path).__dict__[name]


def colmap_poses_to_intrinsics_extrinsics(data):
    mat, close, far = data[:, :-2].reshape((-1, 3, 5)), data[:, -2], data[:, -1]
    T, mat2 = mat[:, :, :-1], mat[:, :, -1:]
    h, w, f = mat2[:, 0, 0], mat2[:, 1, 0], mat2[:, 2, 0]

    # Since we only have width, height and focal point, the intrinsics matrix will give imprecise results
    K = torch.zeros((T.shape[0], 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 2, 2], K[:, 0, 2], K[:, 1, 2] = f, f, 1, w / 2, h / 2

    return K, T, (h, w)


def preprocess_scene_video(video_path, K, R, t, fps):
    # Preprocesses data for a scene video and returns the result. Should be used in dataset.__getitem__()
    # R and t may be either just one matrix for the entire thing (static cameras) or a batch of matrices, one for each frame (moving cameras)
    # TODO do matrix computations from here precached and store in database, especially the ones for moving cameras, which have per-frame matrices
    # TODO add an UV option too (to convert to uv) (also actually add it precached too)
    K, R, t = [i if isinstance(i, torch.Tensor) else torch.tensor(i) for i in (K, R, t)]
    video = VideoDecoder(video_path)
    shape = [len(video), *video[0].shape]

    # Takes into account resized images
    h_real, w_real = shape[-2:]
    h, w = 2 * K[1, 2], 2 * K[0, 2]
    K[1, 2], K[0, 2] = h_real / 2, w_real / 2 # p_y, p_x
    K[0, 0], K[1, 1] = K[0, 0] * (w_real / w), K[1, 1] * (h_real / h) # c_x, c_y
    
    # Frame times
    time = torch.arange(shape[-4]) / fps
    
    return {
        'video': video,
        'K': K,
        'Kinv': K.inverse(),
        'R': R.squeeze().reshape((-1, 3, 3)),
        't': t.squeeze().reshape((-1, 3)),
        'time': time,
        'shape': shape
    }
