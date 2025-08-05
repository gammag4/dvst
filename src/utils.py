import importlib
import math
import random

from easydict import EasyDict as edict

import torch
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.functional as VF


def format_big_number(num):
    if num < 1:
        return f'{num}'

    suffixes = ['', 'K', 'M', 'B', 'T']
    i = min(len(suffixes) - 1, int(math.floor(math.log10(num) / 3)))
    unit = suffixes[i]
    num = num / (10 ** (3 * i))

    return f"{num:.2f}{unit}"


def get_num_params(model, print_params=True):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    res = f'Total params: {format_big_number(total_params)}; Trainable params: {format_big_number(trainable_params)}'
    if print_params:
        print(res)
    else:
        return res


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


def preprocess_scene_video(video_path, K, R, t, fps, resize_to, device):
    # Preprocesses data for a scene video and returns the result
    # R and t may be either just one matrix for the entire thing (static cameras) or a batch of matrices, one for each frame (moving cameras)
    # TODO do matrix computations from here precached and store in database, especially the ones for moving cameras, which have per-frame matrices
    # TODO add an UV option too (to convert to uv) (also actually add it precached too)

    K, R, t = [i.to(device) if isinstance(i, torch.Tensor) else torch.tensor(i, device=device) for i in (K, R, t)]
    video = VideoDecoder(video_path, device=device)
    shape = torch.Size([len(video), *(video[0].shape if resize_to is None else (video[0].shape[0], *resize_to))])

    # Takes into account resized images
    h_real, w_real = shape[-2:] if resize_to is None else resize_to
    h, w = 2 * K[1, 2], 2 * K[0, 2]
    K[1, 2], K[0, 2] = h_real / 2, w_real / 2 # p_y, p_x
    K[0, 0], K[1, 1] = K[0, 0] * (w_real / w), K[1, 1] * (h_real / h) # c_x, c_y
    
    R = R.squeeze().reshape((-1, 3, 3))
    t = t.squeeze().reshape((-1, 3))
    
    R = R.repeat((shape[0], 1, 1)) if R.shape[0] == 1 else R
    t = t.repeat((shape[0], 1)) if t.shape[0] == 1 else t
    
    # Frame times
    time = torch.arange(shape[-4], device=device) / fps
    
    return edict(
        video=video,
        K=K,
        Kinv=K.inverse(),
        R=R,
        t=t,
        time=time,
        shape=shape,
        resize_to=resize_to
    )


def preprocess_scene_videos(scene, device):
    # Processes data from multiple scene video tuples, which should be in the format of the args given to preprocess_scene_video
    # Should be used in dataset.__getitem__()
    
    video_tuples = scene.video_tuples
    n_sources = scene.n_sources
    n_targets = scene.n_targets
    shuffle = scene.shuffle
    shuffle_before_splitting = scene.shuffle_before_splitting
    resize_to = scene.resize_to
    
    videos = [preprocess_scene_video(*v, resize_to, device) for v in video_tuples]
    n_frames = min((v.shape[-4] for v in videos))

    if shuffle and shuffle_before_splitting:
        random.shuffle(videos)

    sources, targets = videos[-n_sources:], videos[:n_targets]

    if shuffle and not shuffle_before_splitting:
        random.shuffle(sources)
        random.shuffle(targets)
    
    targets, queries = targets, targets # TODO fix

    # Sources is the list of source videos that will be used to create latent representation of scene
    # Queries is the list of each frame query (pose + time frame) to be retrieved
    # Targets is the ground truth videos
    return edict(
        sources=sources,
        queries=queries,
        targets=targets,
        n_frames=n_frames,
    )


def get_video_slice(v, start, end):
    res = edict(
        K=v.K,
        Kinv=v.Kinv,
        R=v.R if v.R.shape[0] == 1 else v.R[start:end],
        t=v.t if v.t.shape[0] == 1 else v.t[start:end],
        time=v.time[start:end],
        shape=torch.Size([end - start, *v.shape[1:]])
    )
    
    if v.video is not None:
        video = v.video[start:end] / 255.0
        if v.resize_to is not None:
            video = VF.resize(video, v.resize_to)

        res.video = video
    
    return res


def get_videos_slice(videos, start, end):
    return [get_video_slice(v, start, end) for v in videos]
