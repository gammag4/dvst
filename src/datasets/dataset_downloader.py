import os
from abc import ABC, abstractmethod

from src.utils import json_load, json_dump

class DatasetDownloader(ABC):
    def __init__(self, dataset_name, path):
        self.dataset_name = dataset_name
        self.path = path
    
    @abstractmethod
    def _download(self):
        pass
    
    def download(self):
        check_downloaded_path = os.path.join(self.path, f'check_fully_downloaded.json')
        
        try:
            json_load(check_downloaded_path)
            print(f'Dataset "{self.dataset_name}" already downloaded')
            return
        except:
            pass
        
        self._download()
        
        json_dump(check_downloaded_path, [True])
