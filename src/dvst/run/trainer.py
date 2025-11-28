import gc
from typing import cast
import random
import torch
import torch.distributed as dist
import einx
from easydict import EasyDict as edict

from src.base.run import DistributedTrainer, DefaultDistributedTrainer

from src.dvst.datasets.scene_dataset import SceneDataset, Scene, SceneData, SourceBatch, QueryBatch, SceneBatch
from src.dvst.config import *
from src.dvst.loss import PerceptualLoss
from src.dvst.model import DVST


def compute_params_metrics(params_list):
    if isinstance(params_list, torch.Tensor):
        return torch.stack([params_list.norm(2), params_list.mean(), params_list.std(), params_list.max(), params_list.min()]).to('cpu', non_blocking=True)
    
    norm = torch.stack([p.norm(2) for p in params_list]).square().sum().sqrt()
    
    nels, means, stds = zip(*[[p.numel(), p.mean(), p.std()] for p in params_list])
    means = torch.stack(means)
    stds = torch.stack(stds)
    nels = torch.tensor(nels, device=means[0].device)
    mean = (means * nels).sum() / nels.sum()
    std = ((stds.square() + (means - mean).square()) * nels).sum() / nels.sum()
    
    p_max = torch.stack([p.max() for p in params_list]).max()
    p_min = torch.stack([p.min() for p in params_list]).min()
    
    return torch.stack([norm, mean, std, p_max, p_min]).to('cpu', non_blocking=True)


class DVSTTrainer(DefaultDistributedTrainer[DVSTDatasetConfig, DVSTModelConfig, DVSTOptimizerConfig, DVSTLossConfig, DVST]):
    def load_default_state(self):
        super().load_default_state()
        
        self.last_frames = None
    
    def state_dict(self):
        state_dict = super().state_dict()
        
        state_dict['last_frames'] = self.last_frames
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        
        self.last_frames = state_dict['last_frames']
    
    def _run_forward(self, *args):
        # TODO
        # scene_batch, mask = args
        # # Merges frame and view dims and does masking
        # I, K, R, t, time = [einx.rearrange('b f v ... -> b (f v) ...', k)[:, sources_mask, ...] for k in (I, K, R, t, time)]
        # I, K, R, t, time = [einx.rearrange('b f v ... -> b (f v) ...', k)[:, targets_mask, ...] for k in (I, K, R, t, time)]
        
        
        # Batches shapes: `(b (f v) ...)`
        #   where `f = 1` for now (single frame)
        #   and `b = 1` if stack_similar_batches = False
        batch: list[SceneBatch | None]
        scenes_latents: list[torch.Tensor | None]
        batch, = args
        scenes_latents = [None] * len(batch)
        
        # TODO Whether batches and latents with same sizes should be stacked together to make computations faster
        stack_similar_batches = True
        
        # Stacks batches with same sizes to speed up computations and waste less memory
        if stack_similar_batches:
            batch_dict = {}
            for i, (b, l) in enumerate(zip(batch, scenes_latents)):
                if b is None:
                    continue
                
                # Concatenates batches with same number of frames, number of channels, height, width and latent window size
                k = (*b.sources.I.shape[-4:], *b.targets.I.shape[-4:], None if l is None else l.shape[-2])
                b_list, l_list, i_list = batch_dict.get(k, ([], [], []))
                b_list.append(b), l_list.append(l), i_list.append(i)
                batch_dict[k] = (b_list, l_list, i_list)
            
            b_list, l_list, i_list = zip(*[[b, None if l[0] is None else torch.concat(l), i] for b, l, i in batch_dict.values()])
            sources, targets = [], []
            for b in b_list:
                b_sources, b_targets = zip(*[(b2.sources, b2.targets) for b2 in b])
                b_sources, b_targets = [SourceBatch(*[torch.concat(k2) for k2 in zip(*[[s.K, s.R, s.t, s.time, s.I] for s in k])]) for k in (b_sources, b_targets)]
                sources.append(b_sources), targets.append(b_targets)
            
            scenes_latents = l_list
        else:
            i_list, batch, scenes_latents = zip(*[[i, b, s] for i, (b, s) in enumerate(zip(batch, scenes_latents)) if b is not None])
            sources, targets = zip(*[[b.sources, b.targets] for b in batch])
            scenes_latents = self.scenes_latents
        
        losses, scenes_latents, self.last_frames = self.model(list(sources), list(targets), list(scenes_latents))
        
        losses = [i for l in losses for i in l] if isinstance(losses, list) else losses
        numels = [l.numel() for l in losses]
        self.logger.log({
            'per_image_scene_losses': (torch.stack([l.sum() / n for l, n in zip(losses, numels)]) if isinstance(losses, list) else losses.mean(dim=-1)).detach().to('cpu', non_blocking=True),
            'per_image_loss': ((torch.stack([l.sum() for l in losses]).sum() if isinstance(losses, list) else losses.sum()) / sum(numels)).detach().to('cpu', non_blocking=True),
        })
        loss = torch.stack([l.sum() for l in losses]).sum() if isinstance(losses, list) else losses.sum()
        
        return loss
    
    def _step(self):
        super()._step()
        
        if isinstance(self.base_model.loss, PerceptualLoss):
            loss = cast(PerceptualLoss, self.base_model.loss)
            self.logger.log({'perceptual_weights': loss.layer_weights.to('cpu', non_blocking=True)})
        
        self.logger.log({'optimizer_param_groups_lrs': [p['lr'] for p in self.optimizer.param_groups]})
        
        self.logger.log({
            'iteration_stats': {
                'metrics': ['norm', 'mean', 'std', 'max', 'min'],
                'model': compute_params_metrics([p.detach() for p in self.base_model.parameters() if p.requires_grad]),
                'grad': compute_params_metrics([p.grad.detach() for p in self.base_model.parameters() if p.requires_grad and p.grad is not None]),
                'start_latent_embeds': compute_params_metrics(self.base_model.start_latent_embeds.detach()),
                'last_frame.gen': compute_params_metrics(self.last_frames['gen'][0]),
                'last_frame.target': compute_params_metrics(self.last_frames['target'][0]),
            }
        })
    
    def _run_epoch(self):
        # Keeps getting scenes from two datasets, one with long scenes (to learn to store longer scenes) and another with short scenes (to give more variability and prevent it from overfitting to the longer scenes)
        
        data_it = iter(self.train_data)
        
        n_train_steps = self.config.train.n_train_steps
        batch_size = self.config.train.batch_size
        
        for self.current_batch in range(self.current_batch, n_train_steps):
            scenes: list[Scene | None] = [None] * batch_size
            batch: list[SceneBatch | None] = [None] * batch_size
            
            self.logger.log({ 'batch': self.current_batch })
            
            for i in range(len(scenes)):
                s = next(data_it, None)
                if s is None:
                    # If dataset ends, reiterates over entire dataset
                    data_it = iter(self.train_data)
                    s = next(data_it)
                
                scenes[i] = s.load(self.device)
                batch[i] = scenes[i].get_next_frames(num_frames=1, num_targets_back=1)
            
            self.logger.log({'scene_ids': [f'{s.scene_id}' for s in scenes]})
            
            self._run_pass(batch)
            
            # TODO
            # Saving up memory for next scene
            del scenes
            del batch
            gc.collect()
            # torch.accelerator.memory.empty_cache() # Not needed
        
        self.current_batch = 0
    
    def _train(self):
        print(f'Number of features for each encoded patch in encoder/decoder: {(self.base_model.encoder.pose_encoder.linear.in_features, self.base_model.decoder.pose_encoder.linear.in_features)}')
        
        return super()._train()
