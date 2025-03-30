import os
import json

from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder


class PanopticDataset(Dataset):
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, 'data.json')) as f:
            data = json.load(f)

        for d in data:
            for i in d[1]:
                i['filename'] = os.path.join(path, d[0], i['filename'])

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i][1]
        res = [(VideoDecoder(v['filename']), v['K'], v['R'],
                v['t'], v['fps'], v['shape']) for v in d]
        return res
