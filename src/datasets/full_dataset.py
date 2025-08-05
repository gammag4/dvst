from torch.utils.data import Dataset

from easydict import EasyDict as edict


class FullDataset(Dataset):
    def __init__(self, config):
        self.datasets = [
            edict(
                dataset=c.dataset,
                n_sources=c.n_sources,
                n_targets=c.n_targets,
                shuffle=c.shuffle,
                shuffle_before_splitting=c.shuffle_before_splitting,
                resize_to=c.resize_to
            ) for c in config
        ]
    
    def __len__(self):
        return sum((len(d.dataset) for d in self.datasets))
    
    def __getitem__(self, i):
        for d in self.datasets:
            l = len(d.dataset)
            if i < l:
                return edict(
                    video_tuples=d.dataset[i],
                    n_sources=d.n_sources,
                    n_targets=d.n_targets,
                    shuffle=d.shuffle,
                    shuffle_before_splitting=d.shuffle_before_splitting,
                    resize_to=d.resize_to
                )
            
            i -= l
