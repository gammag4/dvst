import json
import asyncio
import aiohttp
from matplotlib import pyplot as plt
import random
import os
from urllib.request import urlretrieve
import progressbar
from torchcodec.decoders import VideoDecoder
import cv2
import urllib.request


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


class PanopticScene:
    def __init__(self, scene_name):
        self.scene_name = scene_name
        self.url = f'http://domedb.perception.cs.cmu.edu/webdata/dataset/{scene_name}'
        self.video_url = {
            'hd': f'{self.url}/videos/hd_shared_crf20',
            'vga': f'{self.url}/videos/vga_shared_crf10',
            'kinect-color': ''
        }
        self.fps = {'hd': 29.97, 'vga': 25.0, 'kinect-color': 30}

        self.cameras = None

    async def load(self):
        cameras = await self._get_cameras()
        cameras = [self._process_camera(c) for c in cameras]

        self.cameras = {t: list(filter(lambda x: x['type'] == t, cameras)) for t in [
            'hd', 'vga']}

    async def _get_cameras(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{self.url}/calibration_{self.scene_name}.json') as resp:
                return (await resp.json())['cameras']

    def _process_camera(self, c):
        cam_type = c['type']
        cam_name = c['name']
        cam_fname = f'{cam_type}_{cam_name}.mp4'
        
        c['fps'] = self.fps[cam_type]
        c['url'] = f'{self.video_url[cam_type]}/{cam_fname}'
        c['filename'] = cam_fname
        
        return {k: c[k] for k in ['filename', 'type', 'url', 'K', 'R', 't', 'fps']}
        # return c

    def get_random_views(self, cam_type, n_views):
        cameras = self.cameras[cam_type].copy()
        random.shuffle(cameras)
        return cameras[:n_views]


class PanopticDownloader:
    def __init__(self, device):
        self.scenes = None
        self.device = device

    async def load(self, scene_names_file):
        scenes = [PanopticScene(n) for n in self._get_scene_names(scene_names_file)]

        bsize = 3
        batches = [scenes[i:i+bsize] for i in range(0, len(scenes), bsize)]
        for batch in progressbar.progressbar(batches):
            await asyncio.gather(*[s.load() for s in batch])

        self.scenes = scenes

    def _get_scene_names(self, scene_names_file):
        with open(scene_names_file, encoding='utf-8') as f:
            return [i.strip() for i in f.readlines()]

    def download_views(self, path, cam_type, n_views, n_scenes=None):
        scenes = self.scenes.copy()
        if n_scenes is not None:
            random.shuffle(scenes)
            scenes = scenes[:n_scenes]
        scenes = [(s.scene_name, s.get_random_views(cam_type, n_views)) for s in scenes]

        for i, (name, views) in enumerate(progressbar.progressbar(scenes)):
            for j, v in enumerate(views):
                p = os.path.join(path, name)
                os.makedirs(p, exist_ok=True)
                p = os.path.join(p, v['filename'])
                urlretrieve(v['url'], p)

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'data.json'), 'w', encoding='utf-8') as f:
            json.dump(scenes, f, indent=2, sort_keys=True)


# async def main():
#     d = PanopticScene('160224_ultimatum1')
#     await d.load()
#     d.cameras['vga'][0]

#     d = PanopticDownloader()
#     await d.load(scene_names_file='panoptic_scene_names.txt')

# main()
