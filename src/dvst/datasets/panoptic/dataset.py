import os

from src.base.utils import json_load

from src.dvst.datasets.scene_dataset import VideoViewData, SceneData, SceneDataset


class PanopticDataset(SceneDataset):
    def __init__(self, path, resize_to, n_sources, n_targets):
        self.path = path
        self.fps = {'hd': 29.97, 'vga': 25.0, 'kinect-color': 30}
        
        scenes = []
        for sname in os.listdir(self.path):
            if not os.path.isdir(os.path.join(self.path, sname)):
                continue
            
            spath = os.path.join(self.path, sname)
            data = json_load(os.path.join(spath, f'data.json'))
            
            view_datas = [VideoViewData(
                path=os.path.join(spath, v.path),
                K=v.K,
                R=v.R,
                t=v.t,
                time=None,
                fps=v.fps,
                shape=v.shape,
                resize_to=resize_to
            ) for v in data.views]
            
            scene = SceneData.from_sources_targets_split(
                view_datas,
                data.n_frames,
                n_sources,
                n_targets,
                True,
                True
            )
            
            scenes.append(scene)
        
        self.scenes = scenes
        self._n_frames = sum([s.n_frames for s in scenes])
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, i):
        return self.scenes[i]
    
    @property
    def n_frames(self):
        return self._n_frames
    
    @property
    def n_scenes(self):
        return len(self.scenes)
