from torch.utils.data import Dataset

from src.base.config import Config


class FullDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
    
    def __len__(self):
        return sum((len(d) for d in self.datasets))
    
    def __getitem__(self, i):
        for d in self.datasets:
            l = len(d)
            if i < l:
                return d[i]
            
            i -= l
