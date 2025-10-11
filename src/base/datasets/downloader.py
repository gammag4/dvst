import os
from abc import ABC, abstractmethod

from src.base.utils import json_load, json_dump


class DatasetDownloader(ABC):
    def __init__(self, dataset_name, path):
        self.dataset_name = dataset_name
        self.path = path
    
    @abstractmethod
    async def _download(self):
        pass
    
    @abstractmethod
    def _post_process(self):
        pass
    
    async def download(self):
        check_downloaded_path = os.path.join(self.path, f'check_fully_downloaded.json')
        
        should_download = True
        should_process = True
        
        try:
            res = json_load(check_downloaded_path)
            should_download = not res.get('downloaded', False)
            should_process = not res.get('post_processed', False)
        except:
            pass
        
        if should_download:
            print(f'Downloading dataset "{self.dataset_name}"')
            await self._download()
            json_dump(check_downloaded_path, {'downloaded': True, 'post_processed': False})
            print(f'Dataset "{self.dataset_name}" downloaded')
        else:
            print(f'Dataset "{self.dataset_name}" already downloaded')
        
        if should_process:
            print(f'Processing dataset "{self.dataset_name}"')
            self._post_process()
            json_dump(check_downloaded_path, {'downloaded': True, 'post_processed': True})
            print(f'Dataset "{self.dataset_name}" processed')
        else:
            print(f'Dataset "{self.dataset_name}" already processed')
