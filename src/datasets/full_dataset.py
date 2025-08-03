from torch.utils.data import Dataset

from .raw_dataset_wrapper import RawDatasetWrapper


class FullDataset(Dataset):
    def __init__(self, config):
        self.datasets = [
            RawDatasetWrapper(
                dataset=c.dataset,
                shuffle=c.shuffle,
                shuffle_before_splitting=c.shuffle_before_splitting,
                n_sources=c.n_sources,
                n_targets=c.n_targets
            ) for c in config
        ]
        
    def to(self, device):
        for d in self.datasets:
            d.to(device)
        return self
        
    def __len__(self):
        return sum((len(d) for d in self.datasets))
    
    def __getitem__(self, i):
        for d in self.datasets:
            l = len(d)
            if i < l:
                return d[i]
            
            i -= l
