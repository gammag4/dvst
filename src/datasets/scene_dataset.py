import math
import random
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import torch
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2.functional as v2f


# The view can be either VideoDecoder, a tensor shaped (B, C, H, W), or None if it is a query
# TODO dependency injection pass view to decoder
class View:
    def __init__(self, view: VideoDecoder | torch.Tensor | None, K, Kinv, R, t, time, shape, resize_to=None):
        self._view = view
        self.K = K
        self.Kinv = Kinv
        self.R = R
        self.t = t
        self.time = time
        self.shape = shape
        self.resize_to = resize_to
        self._loaded = False
    
    @property
    def view(self):
        # Lazily applies transforms to video
        if self._loaded:
            return self._view
        
        view = self._view
        
        if self.resize_to is not None:
            view = v2f.resize(view, self.resize_to)

        if not view.dtype.is_floating_point:
            view = view / 255.0
        
        self._view = view
        self._loaded = True
        
        return view
    
    def get_slice(self, start, end):
        view = self._view[start:end]
        K = self.K
        Kinv = self.Kinv
        R = self.R if self.R.shape[0] == 1 else self.R[start:end]
        t = self.t if self.t.shape[0] == 1 else self.t[start:end]
        time = self.time[start:end]
        shape = torch.Size([end - start, *self.shape[1:]])
        
        new_view = View(
            view,
            K,
            Kinv,
            R,
            t,
            time,
            shape,
            self.resize_to
        )
        new_view._loaded = self._loaded
        
        return new_view
    
    def as_query(self):
        return View(
            None,
            self.K,
            self.Kinv,
            self.R,
            self.t,
            self.time,
            self.shape,
            self.resize_to
        )
        
    def as_target(self):
        return self._view


class AbstractViewData(ABC):
    # resize_to Tells which dimensions to resize frames to (H, W). If None, does not resize
    # Either time or fps should be set but not both
    def __init__(self, K, R, t, time: torch.Tensor | None, fps: float | torch.FloatType | None, shape, resize_to):
        assert (time is None) ^ (fps is None), 'Either time or fps should be set, but not both'

        self.K = K
        self.R = R
        self.t = t
        self.time = time
        self.fps = fps
        self.shape = shape
        self.resize_to = resize_to
    
    @abstractmethod
    def _load_raw_view(self, device) -> VideoDecoder | torch.Tensor:
        # Has to return an object that has .shape in format (B, C, H, W) and that can be accessed using [b1:b2, c1:c2, h1:h2, w1:w2]
        # TODO default is return VideoDecoder(self.view_path, device=device)
        pass
    
    def load_view(self, device):
        # Preprocesses data for a scene view and returns the result
        # R and t may be either just one matrix for the entire thing (static cameras) or a batch of matrices, one for each frame (moving cameras)
        # TODO do matrix computations from here precached and store in database, especially the ones for moving cameras, which have per-frame matrices
        # TODO add an UV option too (to convert to uv) (also actually add it precached too)

        # View
        view = self._load_raw_view(device)
        shape = torch.Size([len(view), *(view[0].shape if self.resize_to is None else (view[0].shape[0], *self.resize_to))])
        
        # Frame times
        time = self.time if self.fps is None else torch.arange(shape[-4], device=device) / self.fps

        K, R, t = self.K, self.R, self.t
        K, R, t, time = [i.to(torch.float32).to(device) if isinstance(i, torch.Tensor) else torch.tensor(i, device=device) for i in (K, R, t, time)]

        # Takes into account resized images
        h_real, w_real = shape[-2:] if self.resize_to is None else self.resize_to
        h, w = 2 * K[1, 2], 2 * K[0, 2]
        K[1, 2], K[0, 2] = h_real / 2, w_real / 2 # p_y, p_x
        K[0, 0], K[1, 1] = K[0, 0] * (w_real / w), K[1, 1] * (h_real / h) # c_x, c_y
        
        R = R.squeeze().reshape((-1, 3, 3))
        t = t.squeeze().reshape((-1, 3))
        
        R = R.repeat((shape[0], 1, 1)) if R.shape[0] == 1 else R
        t = t.repeat((shape[0], 1)) if t.shape[0] == 1 else t
        
        return View(
            view,
            K,
            K.inverse(),
            R,
            t,
            time,
            shape,
            self.resize_to
        )


class VideoViewData(AbstractViewData):
    def __init__(self, path, K, R, t, time: torch.Tensor | None, fps: float | torch.FloatType | None, shape, resize_to):
        super().__init__(K, R, t, time, fps, shape, resize_to)
        self.path = path
    
    def _load_raw_view(self, device):
        return VideoDecoder(self.path, device=device)


# This may either be an entire scene or just one batch
# Iterating over it iterates over its batches
class Scene:
    def __init__(self, views: list[View], n_frames, sources_idx, queries_targets_idx, batch_size):
        # Sources is the list of source views that will be used to create latent representation of scene
        # Query-target tuples is a list of tuples of each frame query (pose + time frame) to be retrieved and its respective ground truth view
        self._views = views
        self._sources_idx = sources_idx
        self._queries_targets_idx = queries_targets_idx # TODO separate queries and targets
        
        self.sources = [self._views[i] for i in self._sources_idx]
        queries_targets = [self._views[i] for i in self._queries_targets_idx]
        self.query_target_tuples = [(v.as_query(), v.as_target()) for v in queries_targets]
        
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.n_batches = math.ceil(n_frames / batch_size)
        
        self._current = 0
    
    def get_slice(self, start, end):
        return Scene(
            [v.get_slice(start, end) for v in self._views],
            self.n_frames,
            self._sources_idx,
            self._queries_targets_idx,
            self.batch_size
        )
    
    def __iter__(self):
        return self
    
    # Iterates over scene batches
    def __next__(self):
        if self._current >= self.n_batches:
            raise StopIteration

        start = self._current * self.batch_size
        end = start + self.batch_size
        self._current += 1
        return self.get_slice(start, end)


class SceneData:
    def __init__(self, view_datas: list[AbstractViewData], n_frames, sources_idx, queries_targets_idx):
        self.view_datas = view_datas
        self.n_frames = n_frames
        self.sources_idx = sources_idx
        self.queries_targets_idx = queries_targets_idx
    
    def load_scene(self, batch_size, device):
        return Scene(
            [v.load_view(device) for v in self.view_datas],
            self.n_frames,
            self.sources_idx,
            self.queries_targets_idx,
            batch_size
        )
    
    # If n_sources or n_targets are None, all videos are chosen as sources/targets respectively
    @staticmethod
    def from_sources_targets_split(view_datas: list[AbstractViewData], n_frames, n_sources, n_targets, shuffle, shuffle_before_splitting):
        if shuffle and shuffle_before_splitting:
            random.shuffle(view_datas)
        
        idx = list(range(len(view_datas)))
        sources_idx, queries_targets_idx = idx[-n_sources:], idx[:n_targets]

        if shuffle and not shuffle_before_splitting:
            random.shuffle(sources_idx)
            random.shuffle(queries_targets_idx)
        
        return SceneData(
            view_datas,
            n_frames,
            sources_idx,
            queries_targets_idx
        )


class SceneDataset(ABC, Dataset[SceneData]):
    def __init__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, index) -> SceneData:
        pass
    
    @property
    @abstractmethod
    def n_frames(self):
        pass
    
    @property
    @abstractmethod
    def n_scenes(self):
        pass
