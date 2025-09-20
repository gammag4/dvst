import asyncio
import random
import os
import shutil
import progressbar
from easydict import EasyDict as edict

from src.base.utils import json_load, json_dump, json_get, text_get
from src.base.datasets import DatasetDownloader

from src.dvst.utils import ffmpeg_try_process_video, get_video_info


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


def try_remove(fpath):
    try:
        os.remove(fpath)
    except:
        pass


async def run_batched(funcs, batch_size):
    res = []
    batches = [funcs[i:i+batch_size] for i in range(0, len(funcs), batch_size)]
    for batch in batches:
        res.extend(await asyncio.gather(*[f(*args) for f, args in batch]))
    return res


# TODO either download to a shared file system (when distributed) or do lazy download only when using scene for training (but would do multiple requests one for each process so could have rate limiting)
class PanopticDownloader(DatasetDownloader):
    # resize_to: Tuple with width and height to resize to. If None, does not resize, and if either width or height is -1, it resizes maintaining aspect ratio
    # n_scenes: num scenes to download (None means download all)
    # n_views: num hd views to download (None means download all)
    # Example: resize_to=(-1, 256) (resizes to height 256 maintaining aspect ratio), n_scenes=None (all), n_views=8 (8 hd) (max num hd vga kinect videos per scene)
    def __init__(self, path, use_cuda=True, cq_amount=23, resize_to=None, n_scenes=None, n_views=8, seed=42, use_snu_endpoint=False):
        super().__init__('panoptic', path)
        
        self.video_url = {
            'hd': '/videos/hd_shared_crf20',
            'vga': '/videos/vga_shared_crf10',
            'kinect-color': ''
        }
        self.fps = {'hd': 29.97, 'vga': 25.0, 'kinect-color': 30}
        endpoint = 'http://vcl.snu.ac.kr/panoptic' if use_snu_endpoint else f'http://domedb.perception.cs.cmu.edu'
        self.url = f'{endpoint}/webdata/dataset'
        self.resize_to = resize_to
        self.n_scenes = n_scenes
        self.n_views = {'hd': n_views} # TODO change if you want to add other types of videos
        self.rand = random.Random(seed)
        self.scene_names_url = 'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/246ebcfeeeab9d52fb4843b6698abe2d1a66c5e1/scripts/getDB_panoptic_ver1_2.sh'
        
        self.use_cuda = use_cuda
        self.cq_amount = cq_amount
    
    async def _get_scene_names(self):
        scene_names = await text_get(self.scene_names_url)
        return list(set((i.split()[1] for i in scene_names.split('\n') if i.startswith('$curPath/getData.sh'))))
    
    async def _load_scene_data(self, name):
        print(f'Loading scene "{name}"')
        url = f'{self.url}/{name}'
        calib_data = await json_get(f'{url}/calibration_{name}.json')
        
        cameras = edict()
        for c in calib_data['cameras']:
            l = cameras.get(c['type'], [])
            l.append(c)
            cameras[c['type']] = l
        
        for c in cameras.values():
            self.rand.shuffle(c)
        
        return name, url, calib_data, cameras
    
    async def _load_scenes_data(self):
        backup_path = os.path.join(self.path, 'full_data.json')
        
        try:
            return json_load(backup_path)
        except:
            pass
        
        batch_size = 3
        scene_names = await self._get_scene_names()
        scenes = await run_batched([(self._load_scene_data, [n]) for n in scene_names], batch_size)
        self.rand.shuffle(scenes)
        
        json_dump(backup_path, scenes)
        return scenes
    
    # TODO fix this abomination
    async def _download(self):
        backup_path = os.path.join(self.path, 'download_progress.json')
        
        try:
            curr_scene, curr_types, curr_type, curr_cam, downloaded_scenes, downloaded_scams, downloaded_views = json_load(backup_path)
        except:
            curr_scene, curr_types, curr_type, curr_cam, downloaded_scenes, downloaded_scams, downloaded_views = 0, list(self.n_views.keys()), 0, 0, 0, [], 0
                
        os.makedirs(self.path, exist_ok=True)
        scenes = await self._load_scenes_data()
        
        for curr_scene in range(curr_scene, len(scenes)):
            sname, surl, calib_data, scams = scenes[curr_scene]
            spath = os.path.join(self.path, sname)
            os.makedirs(spath, exist_ok=True)
            
            for curr_type in range(curr_type, len(curr_types)):
                type = curr_types[curr_type]
                vids_path = os.path.join(spath, 'videos')
                os.makedirs(vids_path, exist_ok=True)
                
                for curr_cam in range(curr_cam, len(scams[type])):
                    json_dump(backup_path, [curr_scene, curr_types, curr_type, curr_cam, downloaded_scenes, downloaded_scams, downloaded_views])
                    
                    cam = scams[type][curr_cam]
                    fname = f'{type}_{cam['name']}.mp4'
                    vpath = os.path.join(vids_path, fname)
                    vurl = f'{surl}{self.video_url[type]}/{fname}'
                    
                    if not ffmpeg_try_process_video(vurl, vpath, resize_to=self.resize_to, cq_amount=self.cq_amount, use_cuda=self.use_cuda):
                        try_remove(vpath)
                        continue
                    
                    info = get_video_info(vpath)
                    data = edict(
                        path=vpath,
                        K=cam['K'],
                        R=cam['R'],
                        t=cam['t'],
                        dist_coef=cam['distCoef'],
                        fps=info.average_fps,
                        shape=info.shape,
                    )
                    
                    downloaded_scams.append(data)
                    downloaded_views += 1
                    if self.n_views.get(type, None) is not None and downloaded_views == self.n_views[type]:
                        break
                
                downloaded_views = 0
                curr_cam = 0
            
            if len(downloaded_scams) == 0:
                shutil.rmtree(spath, ignore_errors=True)
            else:
                n_frames = max([c['shape'][0] for c in downloaded_scams])
                
                json_dump(os.path.join(spath, f'data.json'), edict(
                    views=downloaded_scams,
                    n_frames=n_frames,
                ))

                downloaded_scenes += 1
                if self.n_scenes is not None and downloaded_scenes == self.n_scenes:
                    break
            
            downloaded_scams = []
            curr_type = 0
        
        total_n_frames = 0
        
        for f in os.listdir(self.path):
            fpath = os.path.join(self.path, f)
            if not os.path.isdir(fpath):
                continue
            
            sdata = json_load(os.path.join(fpath, 'data.json'))
            total_n_frames += sdata.n_frames
        
        json_dump(os.path.join(self.path, f'data.json'), edict(
            n_frames=total_n_frames,
        ))
        
        try_remove(backup_path)
