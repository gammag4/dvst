import math
import random
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import torch
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2.functional as v2f


# TODO organize module into files


class _BatchedIterator:
    def __init__(self, data):
        self.data = data
        self.i = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i >= self.data.n_batches:
            raise StopIteration
        
        batch = self.data.get_batch(self.i)
        self.i += 1
        
        return batch


class _BatchedData(ABC):
    def __init__(self, batch_size=None):
        self.batch_size = batch_size
    
    def __iter__(self):
        return _BatchedIterator(self)
    
    @property
    @abstractmethod
    def _n_items(self):
        pass
    
    @property
    def n_batches(self):
        return math.ceil(self._n_items / self.batch_size)
    
    @abstractmethod
    def get_slice(self, start, end):
        pass
    
    def get_batch(self, i):
        if self.batch_size is None:
            return self.get_slice(None, None)
        
        start = i * self.batch_size
        end = start + self.batch_size
        return self.get_slice(start, end)


class _ViewBase(_BatchedData):
    def __init__(self, view: VideoDecoder | torch.Tensor | None, shape, batch_size=None, resize_to=None, start=None, end=None):
        super().__init__(batch_size)
        self._view = view
        self.shape = shape
        self.resize_to = resize_to
        self._loaded = view is None
        
        self.start = 0 if start is None else start
        self.end = self.shape[0] if end is None else end
    
    @property
    def view(self):
        # Lazily applies transforms to video
        if self._loaded:
            return self._view
        
        view = self._view[self.start:self.end]
        
        if self.resize_to is not None:
            view = v2f.resize(view, self.resize_to)

        if not view.dtype.is_floating_point:
            view = view / 255.0
        
        self._view = view
        self._loaded = True
        
        return view
    
    @property
    def _n_items(self):
        return self.end - self.start
    
    def get_slice(self, start, end):
        shape = torch.Size([end - start, *self.shape[1:]])
        
        start = self.start + start
        end = self.start + end
        end = max(start, min(self.end, end))
        
        view = self._view[start:end] if (self._view is not None) and self._loaded else self._view
        
        return _ViewBase(
            view,
            shape,
            self.batch_size,
            self.resize_to,
            start,
            end
        )


class TargetView(_ViewBase):
    def __init__(self, view: VideoDecoder | torch.Tensor, shape, batch_size=None, resize_to=None, start=None, end=None):
        super().__init__(view, shape, batch_size, resize_to, start, end)
    
    def get_slice(self, start, end):
        s = super().get_slice(start, end)
        new_view = TargetView(
            s._view,
            s.shape,
            s.batch_size,
            s.resize_to,
            s.start,
            s.end
        )
        new_view._loaded = self._loaded
        
        return new_view


# The view can be either VideoDecoder, a tensor shaped (B, C, H, W), or None if it is a query
# TODO dependency injection pass view to decoder
class View(_ViewBase):
    def __init__(self, view: VideoDecoder | torch.Tensor | None, K, Kinv, R, t, time, shape, batch_size=None, resize_to=None, start=None, end=None):
        super().__init__(view, shape, batch_size, resize_to, start, end)
        self.K = K
        self.Kinv = Kinv
        self.R = R
        self.t = t
        self.time = time
    
    def get_slice(self, start, end):
        K = self.K
        Kinv = self.Kinv
        R = self.R if self.R.shape[0] == 1 else self.R[start:end]
        t = self.t if self.t.shape[0] == 1 else self.t[start:end]
        time = self.time[start:end]
        
        s = super().get_slice(start, end)
        
        new_view = View(
            s._view,
            K,
            Kinv,
            R,
            t,
            time,
            s.shape,
            s.batch_size,
            s.resize_to,
            s.start,
            s.end
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
            self.batch_size,
            self.resize_to,
            self.start,
            self.end
        )
    
    def as_target(self):
        target = TargetView(
            self._view,
            self.shape,
            self.batch_size,
            self.resize_to,
            self.start,
            self.end
        )
        target._loaded = self._loaded
        
        return target


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
    
    def load_view(self, batch_size, device):
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
            batch_size,
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
class Scene(_BatchedData):
    def __init__(self, views: list[View], n_frames, batch_size, sources_idx, queries_targets_idx=[], start=None, end=None):
        super().__init__(batch_size)
        # Sources is the list of source views that will be used to create latent representation of scene
        # Queries is a list of frame queries (pose + time frame) to be retrieved
        # Targets is a list of the respective ground truth views for each query
        self._views = views
        self._sources_idx = sources_idx
        self._queries_targets_idx = queries_targets_idx # TODO separate queries and targets
        
        self._sources = None
        self._queries = None
        self._targets = None
        
        self.n_frames = n_frames
        
        self.start = 0 if start is None else start
        self.end = self.n_frames if end is None else end
        
    @property
    def sources(self):
        if self._sources is None:
            self._sources = [self._views[i] for i in self._sources_idx]
        
        return self._sources
        
    @property
    def queries(self):
        if self._queries is None:
            self._queries = [self._views[i].as_query() for i in self._queries_targets_idx]
        
        return self._queries
        
    @property
    def targets(self):
        if self._targets is None:
            self._targets = [self._views[i].as_target() for i in self._queries_targets_idx]
        
        return self._targets
    
    @property
    def _n_items(self):
        return self.n_frames
    
    def get_slice(self, start, end):
        return Scene(
            [v.get_slice(start, end) for v in self._views],
            end - start,
            self.batch_size,
            self._sources_idx,
            self._queries_targets_idx
        )


class SceneData:
    def __init__(self, view_datas: list[AbstractViewData], n_frames, sources_idx, queries_targets_idx):
        self.view_datas = view_datas
        self.n_frames = n_frames
        self.sources_idx = sources_idx
        self.queries_targets_idx = queries_targets_idx
    
    def load_scene(self, batch_size, device):
        return Scene(
            [v.load_view(batch_size, device) for v in self.view_datas],
            self.n_frames,
            batch_size,
            self.sources_idx,
            self.queries_targets_idx
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
