from torch.utils.data import Dataset

from easydict import EasyDict as edict


class FullDataset(Dataset):
    def __init__(self, config):
        self.datasets = config
    
    def __len__(self):
        return sum((len(d) for d in self.datasets))
    
    def __getitem__(self, i):
        for d in self.datasets:
            l = len(d)
            if i < l:
                return d[i]
            
            i -= l
