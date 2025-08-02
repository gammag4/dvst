import os
import json

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

from src.utils import preprocess_scene_video


#TODO create test dataset
class PanopticDataset(Dataset):
    # if is_processed, uses videos with name hd_**_**_r.mp4 instead of hd_**_**.mp4
    def __init__(self, path, is_processed=True):
        self.path = path
        self.is_processed = is_processed
        self.fps = {'hd': 29.97, 'vga': 25.0, 'kinect-color': 30}

        scenes = []
        for sname in os.listdir(path):
            spath = os.path.join(path, sname)
            cal_path = os.path.join(spath, f'calibration_{sname}.json')
            vids_path = os.path.join(spath, 'hdVideos')
            vid_names = list(map(lambda vid: '_'.join(vid.split('_')[1:3]), os.listdir(vids_path)))
            
            with open(cal_path) as f:
                cameras = json.load(f)['cameras']
            
            cameras = {c['name']: c for c in cameras}
            cameras = [cameras[c] for c in vid_names]
            videos = [[
                os.path.join(vids_path, self.get_video_name(c['name'])),
                c['K'],
                c['R'],
                c['t'],
                self.fps[c['type']],
                # c['distCoef'] #TODO currently distortion is being ignored
                ] for c in cameras]
        
            scenes.append(videos)

        self.data = scenes
            
    def get_video_name(self, vid_id):
        return f'hd_{vid_id}_r.mp4' if self.is_processed else f'hd_{vid_id}.mp4'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        return [preprocess_scene_video(*v) for v in d]
