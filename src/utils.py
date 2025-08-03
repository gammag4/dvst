import importlib
import math

from easydict import EasyDict as edict

import torch
from torchcodec.decoders import VideoDecoder


def format_big_number(num):
    if num < 1:
        return f'{num}'

    suffixes = ['', 'K', 'M', 'B', 'T']
    i = min(len(suffixes) - 1, int(math.floor(math.log10(num) / 3)))
    unit = suffixes[i]
    num = num / (10 ** (3 * i))

    return f"{num:.2f}{unit}"


def print_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total params: {format_big_number(total_params)}; Trainable params: {format_big_number(trainable_params)}')


def create_bound_function(self, func):
    # Binds an external function to a class instance as if it was defined in the class
    def new_func(*args, **kwargs): return func(self, *args, **kwargs)
    return new_func


def import_object(full_name):
    # Use [(obj), 'path.to.Object'] to return object
    s = full_name.split('.')
    path, name = '.'.join(s[:-1]), s[-1]
    return importlib.import_module(path).__dict__[name]


def import_and_run_object(full_name, *args):
    obj = import_object(full_name)

    # Use [(obj), 'path.to.Object', null] to run without arguments
    if len(args) == 1 and args[0] is None:
        return obj()
    # Use [(obj), 'path.to.Object', arg1, arg2, ...] to run with arguments
    elif len(args) >= 1:
        return obj(*args)
    # Use [(obj), 'path.to.Object'] to return without running
    else:
        return obj


def colmap_poses_to_intrinsics_extrinsics(data):
    mat, close, far = data[:, :-2].reshape((-1, 3, 5)), data[:, -2], data[:, -1]
    T, mat2 = mat[:, :, :-1], mat[:, :, -1:]
    h, w, f = mat2[:, 0, 0], mat2[:, 1, 0], mat2[:, 2, 0]

    # Since we only have width, height and focal point, the intrinsics matrix will give imprecise results
    K = torch.zeros((T.shape[0], 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 2, 2], K[:, 0, 2], K[:, 1, 2] = f, f, 1, w / 2, h / 2

    return K, T, (h, w)


def preprocess_scene_video(video_path, K, R, t, fps, device):
    # Preprocesses data for a scene video and returns the result
    # R and t may be either just one matrix for the entire thing (static cameras) or a batch of matrices, one for each frame (moving cameras)
    # TODO do matrix computations from here precached and store in database, especially the ones for moving cameras, which have per-frame matrices
    # TODO add an UV option too (to convert to uv) (also actually add it precached too)

    K, R, t = [i.to(device) if isinstance(i, torch.Tensor) else torch.tensor(i, device=device) for i in (K, R, t)]
    video = VideoDecoder(video_path, device=device)
    shape = torch.Size([len(video), *video[0].shape])

    # Takes into account resized images
    h_real, w_real = shape[-2:]
    h, w = 2 * K[1, 2], 2 * K[0, 2]
    K[1, 2], K[0, 2] = h_real / 2, w_real / 2 # p_y, p_x
    K[0, 0], K[1, 1] = K[0, 0] * (w_real / w), K[1, 1] * (h_real / h) # c_x, c_y
    
    # Frame times
    time = torch.arange(shape[-4], device=device) / fps
    
    return edict(
        video=video,
        K=K,
        Kinv=K.inverse(),
        R=R.squeeze().reshape((-1, 3, 3)),
        t=t.squeeze().reshape((-1, 3)),
        time=time,
        shape=shape
    )


def preprocess_scene_videos(video_tuples, device):
    # Processes data from multiple scene video tuples, which should be in the format of the args given to preprocess_scene_video
    # Should be used in dataset.__getitem__()
    videos = [preprocess_scene_video(*v, device) for v in video_tuples]
    n_frames = min((v.shape[-4] for v in videos))

    return edict(
        videos=videos,
        n_frames=n_frames
    )
