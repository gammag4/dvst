import random
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import torch
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2.functional as v2f


def process_view(view, resize_to):
    if resize_to is not None:
        view = v2f.resize(view, resize_to)

    if not view.dtype.is_floating_point:
        view = view / 255.0
    
    return view


def process_K(K, hw):
    # Takes into account resized images
    h_real, w_real = hw
    h, w = 2 * K[..., 1, 2], 2 * K[..., 0, 2]
    K[..., 1, 2], K[..., 0, 2] = h_real / 2, w_real / 2 # p_y, p_x
    K[..., 0, 0], K[..., 1, 1] = K[..., 0, 0] * (w_real / w), K[..., 1, 1] * (h_real / h) # c_x, c_y
    
    return K


class QueryBatch:
    def __init__(self, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, time: torch.Tensor, hw: tuple[int, int] | torch.Size):
        self.K = K
        self.R = R
        self.t = t
        self.time = time
        self.hw = hw


class SourceBatch:
    def __init__(self, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, time: torch.Tensor, I: torch.Tensor):
        self.K = K
        self.R = R
        self.t = t
        self.time = time
        self.I = I


class SceneBatch:
    def __init__(self, sources: SourceBatch, targets: SourceBatch, start_frame: int, end_frame: int, is_last: bool):
        self.sources = sources # I K R t time
        self.targets = targets # I K R t time
        self.is_last = is_last
        self.start_frame = start_frame
        self.end_frame = end_frame
    
    @staticmethod
    def rand_sampling_from_tensors(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, time: torch.Tensor, I: torch.Tensor | list[VideoDecoder], n_sources: int, n_target_frames: int, start_frame: int, end_frame: int, num_targets_back: int, is_last: bool, resize_to: tuple[int, int] | None, sources_perm=None):
        # Randomly selects views
        num_views = I.shape[0] if isinstance(I, torch.Tensor) else len(I)
        sources_perm = torch.randperm(num_views) if sources_perm is None else sources_perm
        
        # Only gets views from the current as source
        sources_slice = slice(None if n_sources is None else -n_sources, None, None)
        K2, R2, t2, time2 = [k[:, start_frame:end_frame, ...][sources_perm][sources_slice] for k in (K, R, t, time)]
        
        if isinstance(I, torch.Tensor):
            I2 = I[:, start_frame:end_frame, ...][sources_perm][sources_slice]
        else:
            I2 = torch.stack([i[start_frame:end_frame] for i in I])[sources_perm][sources_slice]
        I2 = process_view(I2, resize_to)
        
        K2, R2, t2, time2, I2 = [k.permute(1, 0, *range(2, len(k.shape))).flatten(0, 1).unsqueeze(0) for k in (K2, R2, t2, time2, I2)]
        sources = SourceBatch(K2, R2, t2, time2, I2)
        
        # Randomly chooses views from any position starting at `end_frame - num_targets_back` up to the current frame as targets
        # TODO v = 100; torch.rand(3600*30*10, v).topk(v, dim=1).indices
        num_targets_back = min(num_targets_back, end_frame)
        frames = torch.randperm(num_views * num_targets_back)[:n_target_frames]
        views, frames = frames % num_views, frames // num_views + end_frame - num_targets_back
        K2, R2, t2, time2 = [k[[views, frames]] for k in (K, R, t, time)]
        
        if isinstance(I, torch.Tensor):
            I2 = I[[views, frames]]
        else:
            I2 = torch.stack([I[int(v)][int(f)] for v, f in zip(views, frames)])
        I2 = process_view(I2, resize_to)
        
        K2, R2, t2, time2, I2 = [k.unsqueeze(0) for k in (K2, R2, t2, time2, I2)]
        targets = SourceBatch(K2, R2, t2, time2, I2)
        
        # Creates frame slices
        # Shapes: (b (f v) ...) where `b = 1`
        return SceneBatch(
            sources=sources,
            targets=targets,
            start_frame=start_frame,
            end_frame=end_frame,
            is_last=is_last
        )
    
    # @property
    # def n_frames(self):
    #     return self.end_frame - self.start_frame


class Scene(ABC):
    def __init__(self, dataset_name: str, scene_name: str):
        self.scene_id = f'{dataset_name}_{scene_name}'
    
    @abstractmethod
    def get_next_frames(self, num_frames: int = 1, num_targets_back: int = 1) -> SceneBatch:
        pass


class DefaultScene(Scene, ABC):
    def __init__(self, dataset_name: str, scene_name: str, n_frames: int):
        super().__init__(dataset_name, scene_name)
        self.current_frame = 0
        self.n_frames =  n_frames
    
    @abstractmethod
    def _unsafe_get_next_frames(self, start_frame: int, end_frame: int, num_targets_back: int, is_last: bool) -> SceneBatch:
        pass
    
    def get_next_frames(self, num_frames: int = 1, num_targets_back: int = 1):
        end_frame = min(self.n_frames, self.current_frame + num_frames)
        next_frames = self._unsafe_get_next_frames(self.current_frame, end_frame, num_targets_back, self.n_frames == end_frame)
        self.current_frame = end_frame
        return next_frames


class StaticScene(DefaultScene):
    def __init__(self, dataset_name: str, scene_name: str, scene_batch: SceneBatch):
        super().__init__(dataset_name, scene_name, 1)
        self.scene_batch = scene_batch
    
    def _unsafe_get_next_frames(self, start_frame, end_frame, num_targets_back, is_last):
        return self.scene_batch
    
    @staticmethod
    def from_tensors(dataset_name: str, scene_name: str, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, time: torch.Tensor, I: torch.Tensor, n_sources: int, n_target_frames: int, resize_to: tuple[int, int] | None):
        return StaticScene(
            dataset_name=dataset_name,
            scene_name=scene_name,
            scene_batch=SceneBatch.rand_sampling_from_tensors(
                K=K,
                R=R,
                t=t,
                time=time,
                I=I,
                n_sources=n_sources,
                n_target_frames=n_target_frames,
                start_frame=0,
                end_frame=1,
                num_targets_back=1,
                is_last=True,
                resize_to=resize_to
            )
        )


class VideoDecoderScene(DefaultScene):
    def __init__(self, dataset_name: str, scene_name: str, n_frames: int, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, time: torch.Tensor, I: list[VideoDecoder], n_sources: int, n_target_frames: int, resize_to: tuple[int, int] | None):
        super().__init__(dataset_name, scene_name, n_frames)
        self.K = K # v f 3 3
        self.R = R # v f 3 3
        self.t = t # v f 3
        self.time = time # v f
        self.I = I # [v] f c h w
        self.n_sources = n_sources
        self.n_target_frames = n_target_frames
        self.resize_to = resize_to
        self.sources_perm = torch.randperm(len(I)) # Uses same randomly selected views across entire scene
    
    def _unsafe_get_next_frames(self, start_frame, end_frame, num_targets_back, is_last):
        return SceneBatch.rand_sampling_from_tensors(
            K=self.K,
            R=self.R,
            t=self.t,
            time=self.time,
            I=self.I,
            n_sources=self.n_sources,
            n_target_frames=self.n_target_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            num_targets_back=num_targets_back,
            is_last=is_last,
            resize_to=self.resize_to,
            sources_perm=self.sources_perm
        )


class SceneData(ABC):
    @abstractmethod
    def load(self, device) -> Scene:
        pass


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
    
    @abstractmethod
    def get_n_batches(self, batch_size: int | None):
        pass


# TODO maybe decouple this into simple collection for generic datasets?
class CollectionSceneDataset(SceneDataset):
    def __init__(self, datasets: list[SceneDataset]):
        self.datasets = datasets
    
    @property
    def n_frames(self):
        return sum((d.n_frames for d in self.datasets))
    
    @property
    def n_scenes(self):
        return sum((d.n_scenes for d in self.datasets))
    
    def get_n_batches(self, batch_size):
        sum([d.get_n_batches(batch_size) for d in self.datasets])
    
    def __len__(self):
        return sum((len(d) for d in self.datasets))
    
    def __getitem__(self, i):
        for d in self.datasets:
            l = len(d)
            if i < l:
                return d[i]
            
            i -= l
