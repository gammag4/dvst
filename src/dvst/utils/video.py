import math
from easydict import EasyDict as edict
import torch
from torchcodec.decoders import VideoDecoder

from src.base.utils import try_run_cmd


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
