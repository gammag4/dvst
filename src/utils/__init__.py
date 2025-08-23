import os
import importlib
import math
import random
import json
import subprocess
import cgi
from urllib.request import urlopen, urlretrieve
from urllib.error import HTTPError

import aiohttp
from easydict import EasyDict as edict

import torch
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2.functional as v2f


def create_bound_function(self, func):
    # Binds an external function to a class instance as if it was defined in the class
    def new_func(*args, **kwargs): return func(self, *args, **kwargs)
    return new_func


def json_load(path, use_edict=True):
    with open(path, 'r', encoding='utf-8') as f:
        res = json.load(f)
        if use_edict:
            if type(res) is list:
                return edict({'a': res}).a
            else:
                return edict(res)
        else:
            return res


def json_dump(path, data: edict | dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)


async def json_get(url, use_edict=True):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            res = await resp.json()
            return edict(res) if use_edict else res


def try_run_cmd(cmd, verbose=False, raise_err=False):
    # If raise_err, raises if error and only returns output if succeeds
    if verbose:
        print(f'Running "{cmd}"')
    
    if raise_err:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            shell=True
        )
        return res.stdout
    
    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            shell=True
        )
    except subprocess.CalledProcessError as e:
        if verbose:
            print('Error:')
            print(e.stderr)
            print('Command output:')
            print(e.stdout)
        return False, None
    except Exception as e:
        if verbose:
            print(f'Error: {e}')
        return False, None

    return True, res.stdout


def format_big_number(num):
    if num < 1:
        return f'{num}'

    suffixes = ['', 'K', 'M', 'B', 'T']
    i = min(len(suffixes) - 1, int(math.floor(math.log10(num) / 3)))
    unit = suffixes[i]
    num = num / (10 ** (3 * i))

    return f"{num:.2f}{unit}"


def get_num_params(model, print_params=True):
    # TODO change to return model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    res = f'Total params: {format_big_number(total_params)}; Trainable params: {format_big_number(trainable_params)}'
    if print_params:
        print(res)
    else:
        return res


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


# src may be url or path
# resize_to: Tuple with width and height to resize to. If None, does not resize, and if either width or height is -1, it resizes maintaining aspect ratio
# If extracting frames, specify frames to extract as `frame_seconds_to_extract`, a list of times of frames in seconds, and `dst` should be just the base name without extension
def ffmpeg_try_process_video(src, dst, resize_to=None, cq_amount=23, use_cuda=True, show_stats=False, frame_seconds_to_extract=None):
    stats = '-stats' if show_stats else ''
    cuda_params = '-hwaccel cuda -hwaccel_output_format cuda' if use_cuda else ''
    
    resize_filter = 'scale_npp' if use_cuda else 'scale'
    resize_params = '' if resize_to is None else f'-vf {resize_filter}={resize_to[0]}:{resize_to[1]}'
    
    encoder = 'av1_nvenc' if use_cuda else 'libsvtav1'
    
    frames_to_extract = '' if frame_seconds_to_extract is None else f'select={[f'eq(t,{s})' for s in frame_seconds_to_extract]}'
    dst = dst if frame_seconds_to_extract is None else f"{dst}_%06d.png"
    
    # TODO change fps_mode cfr to passthrough and in VideoDecoder get actual frame times for each frame
    # but since that would give variable frame time in some sources, the frames of all videos would need to be sorted by the time they appear in the scene
    # and would also need to change the model to learn variable frame times
    cmd = f'ffmpeg -loglevel quiet {stats} -y {cuda_params} -i "{src}" {resize_params} -map 0:v -c:v {encoder} -cq:v {cq_amount} -fps_mode cfr "{dst}"'

    print(f'Getting and processing video "{dst}"')
    success, _ = try_run_cmd(cmd, verbose=True)

    return success


def get_video_info_old(path):
    # path may be local path or url
    cmd = f'ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=height,width,pix_fmt,nb_read_frames -of default=noprint_wrappers=1:nokey=1 "{path}"'
    out = try_run_cmd(cmd, raise_err=True)
    print(cmd)
    print(out)
    height, width, pix_fmt, n_frames = out.split()

    cmd = f'ffmpeg -hide_banner -pix_fmts | sed "1,/^-----/d" | grep -w "{pix_fmt}"'
    _, _, n_channels, bpp, bit_depths = try_run_cmd(cmd, raise_err=True).split()
    bit_depths = [int(i) for i in bit_depths.split('-')]
    n_bytes_per_pixel = math.ceil(max(bit_depths) / 8)
    
    shape = [n_frames, n_channels, height, width]
    n_bytes_per_frame = n_bytes_per_pixel * n_channels * height * width
    n_bytes_total = n_bytes_per_frame * n_frames
    
    return edict(
        shape=shape,
        n_bytes_per_pixel=n_bytes_per_pixel,
        n_bytes_per_frame=n_bytes_per_frame,
        n_bytes_total=n_bytes_total,
        pix_fmt=pix_fmt,
        bpp=bpp,
        bit_depths=bit_depths,
    )


def get_video_info(path):
    video = VideoDecoder(path)
    first_frame = video[0]
    
    n_frames = len(video)
    shape = [n_frames, *first_frame.shape]
    average_fps = video.metadata.average_fps
    
    n_bytes_per_channel = torch.tensor([], dtype=first_frame.dtype).element_size()
    n_bytes_per_frame = n_bytes_per_channel * first_frame.numel()
    n_bytes_total = n_bytes_per_frame * n_frames
    
    del video
    
    return edict(
        shape=shape,
        average_fps=average_fps,
        n_bytes_per_channel=n_bytes_per_channel,
        n_bytes_per_frame=n_bytes_per_frame,
        n_bytes_total=n_bytes_total,
    )


def colmap_poses_to_intrinsics_extrinsics(data):
    # Format description at github.com/Fyusion/LLFF?tab=readme-ov-file#using-your-own-poses-without-running-colmap
    mat, close, far = data[:, :-2].reshape((-1, 3, 5)), data[:, -2], data[:, -1]
    T, mat2 = mat[:, :, :-1], mat[:, :, -1:]
    h, w, f = mat2[:, 0, 0], mat2[:, 1, 0], mat2[:, 2, 0]

    # Since we only have width, height and focal point, the intrinsics matrix will give imprecise results
    K = torch.zeros((T.shape[0], 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 2, 2], K[:, 0, 2], K[:, 1, 2] = f, f, 1, w / 2, h / 2

    return K, T, (h, w)


def get_path_size(path):
    # Neither os or shutil give accurate results
    # TODO add windows equivalent
    return int(try_run_cmd(f'du -sb "{path}"', raise_err=True).split()[0])


def download_file_maintain_extension(url, path):
    assert len(path.split('.')) == 1, 'Path should not have extension'
    
    try:
        
        remotefile = urlopen(url)
        contentdisposition = remotefile.info()['Content-Disposition']
        _, params = cgi.parse_header(contentdisposition)
        filename = params["filename"]
        urlretrieve(url, filename)
        
        if verbose:
            print(f'Getting file "{url}"')
        
        path = os.path.join(path, os.path.split(url))
        urlretrieve(self.url, path)
        self.path = path
    except HTTPError as e:
        if verbose:
            print(f'Error: {e}')
        return False
    
    return True
