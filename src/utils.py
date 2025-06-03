import torch
from torchcodec.decoders import VideoDecoder

# Preprocesses data for a scene video and returns the result. Should be used in dataset.__getitem__()
# K, R and t may be either just one matrix for the entire thing (static cameras) or a batch of matrices, one for each frame (moving cameras)


def preprocess_scene_video(video_path, K, R, t, fps):
    K, R, t = [i if isinstance(i, torch.Tensor) else torch.tensor(i) for i in (K, R, t)]
    video = VideoDecoder(video_path)
    return {
        'video': video,
        'K': K,
        'R': R,
        't': t,
        'fps': fps,
        'shape': [len(video), *video[0].shape]
    }
