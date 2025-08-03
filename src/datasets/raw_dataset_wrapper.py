import random

from easydict import EasyDict as edict

from torch.utils.data import Dataset


class RawDatasetWrapper(Dataset):
    # If shuffle, it shuffles each scene data
    # If shuffle_before_splitting, it shuffles all scene videos before splitting and shuffles sources and target videos only after splitting else
    # Here, the first 'n_targets' videos in each scene should be the targets and the last 'n_sources' should be the sources (they may overlap)
    #   Always let some videos be only targets to make it learn to generate unseen views or if evaluating the model
    #   If n_targets is None, all cams become targets
    def __init__(self, dataset, shuffle=True, shuffle_before_splitting=True, n_sources=2, n_targets=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.shuffle_before_splitting = shuffle_before_splitting
        self.n_sources = n_sources
        self.n_targets = n_targets
        
    def to(self, device):
        self.dataset.to(device)
        return self
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        s = self.dataset[i]
        videos, n_frames = s.videos, s.n_frames

        if self.shuffle and self.shuffle_before_splitting:
            random.shuffle(videos)

        sources, targets = videos[-self.n_sources:], videos[:self.n_targets]

        if self.shuffle and not self.shuffle_before_splitting:
            random.shuffle(sources)
            random.shuffle(targets)
        
        targets, queries = [v.pop('video') for v in targets], targets

        # Sources is the list of source videos that will be used to create latent representation of scene
        # Queries is the list of each frame query (pose + time frame) to be retrieved
        # Targets is the ground truth videos
        return edict(
            sources=sources,
            queries=queries,
            targets=targets,
            n_frames=n_frames,
        )
