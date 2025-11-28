import os
import shutil
import torch
from torchvision.utils import save_image

checkpoint_file = 'res/tmp/checkpoint/train_data.pt'
frames_dir = 'res/tmp/last_frames'

a = torch.load(checkpoint_file, map_location='cpu')
batch_size, n_frames = a['last_frames']['gen'][0].shape[:2]

if os.path.isdir(frames_dir):
    shutil.rmtree(frames_dir)
os.makedirs(frames_dir, exist_ok=True)

for b in range(batch_size):
    for f in range(n_frames):
        save_image(a['last_frames']['gen'][0][b, f], f'{frames_dir}/{b}_{f}_gen.png')
        save_image(a['last_frames']['target'][0][b, f], f'{frames_dir}/{b}_{f}_target.png')
