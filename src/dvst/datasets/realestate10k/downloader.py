
import asyncio
import aiohttp
import math
import random
import os
import shutil
import progressbar
from urllib.request import urlretrieve
from urllib.error import HTTPError
from easydict import EasyDict as edict
import torch
from torchvision.io import decode_image

from src.base.utils import json_load, json_dump, json_get, text_get
from src.base.datasets import DatasetDownloader

from src.dvst.utils import ffmpeg_try_process_video, get_video_info


# RealEstate10K Dataset: https://google.github.io/realestate10k/download.html
# Downloads the pixelSplat preprocessed version of RE10K described here: https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets
class PixelSplatRealEstate10KDownloader(DatasetDownloader):
    def __init__(self, path):
        self.path = path
    
    async def _download(self):
        pass
    
    def _post_process(self):
        scenes = []
        total_n_frames = 0
        
        files = [i for i in os.listdir(self.path) if i.endswith('.torch')]
        files.sort()
        
        for f in files:
            fpath = os.path.join(self.path, f)
            for i, s in enumerate(torch.load(fpath)):
                n_frames = s['timestamps'].shape[0]
                
                scenes.append(edict(
                    file=f,
                    file_index=i,
                    n_frames=n_frames
                ))
                
                total_n_frames += n_frames
        
        json_dump(os.path.join(self.path, 'data.json'), edict(
            scenes=scenes,
            total_n_frames=total_n_frames
        ))

# TODO incomplete
# RealEstate10K Dataset: https://google.github.io/realestate10k/download.html
# Downloads the raw dataset and preprocesses it
class RealEstate10KDownloader:
    def __init__(self, trajectories_path, path, use_cuda=True, cq_amount=23, resize_to=None, n_scenes=None, seed=42):
        self.trajectories_path = trajectories_path
        self.path = path
    
    def _get_time_string_from_timestamp(self, timestamp):
        ms = round(timestamp * 1000)
        s, ms = math.floor(ms / 1000), ms % 1000
        m, s = math.floor(s / 60), s % 60
        h, m = math.floor(m / 60), m % 60
        return f'{h}:{m}:{s}.{ms}'
    
    def _get_pose_data(self, line):
        data = line.split()
        time, data = int(data[0]) / 1000000, [float(i) for i in data[1:]]
        fx, fy, px, py, data = data[:4], data[6:] # data[4:6] is basically zeros for some reason i dont know
        K = torch.tensor([[fx, 0, px], [0, fy, py], [0, 0, 1]])
        T = torch.tensor(data).reshape((3, 4))
        R, t = T[:, :3], T[:, 3:]

        return time, K, R, t
    
    def _get_scene_data(self, path):
        # Format description: https://google.github.io/realestate10k/download.html
        with open(path, mode='r', encoding='utf-8') as f:
            data = [l.strip() for l in f.readlines()]
        
        video_url, data = data[0], data[1:]
        
        time, K, R, t = list(zip(*[self._get_pose_data(l) for l in data]))
        K, R, t = [torch.stack(i) for i in (K, R, t)]
        
        return edict(
            video_url=video_url,
            time=time,
            K=K,
            R=R,
            t=t,
        )
    
    def _post_process(self):
        pass
