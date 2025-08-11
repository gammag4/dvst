import os
import random
import datetime
import gc

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

import torch.cuda
import torch.multiprocessing as mp # Wrapper around python's native multiprocessing
from torch.utils.data.distributed import DistributedSampler # Model that takes in data and distributes across GPUs
from torch.nn.parallel import DistributedDataParallel as DDP # DDP wrapper
import torch.distributed as dist
import torch.amp as amp

from src.config import load_config
from src.datasets.full_dataset import FullDataset
from src.model import DVST
from src.utils import create_bound_function, preprocess_scene_videos, get_num_params


# Manages gradient scaling, skipping and batch retrying if skipped, and does gradient skipping even with GradScaler disabled
class GradManager(amp.GradScaler):
    # Gradient Scaler with batch retrying
    def __init__(self, device, enabled, batch_retry_enabled, max_retries):
        super().__init__(device=device, enabled=enabled)
        self.batch_retry_enabled = batch_retry_enabled
        self.max_retries = max_retries
        self.should_retry_batch = False
        self.num_retries = 0
        self._skipped = False
    
    def update(self, new_scale = None):
        old_scale = self.get_scale()
        super().update(new_scale)
        
        # Checks whether current batch has been skipped due to inf/nan grads
        # If the scale is smaller than before, it means that it updated its scale because of inf/nan grads
        if self._enabled:
            self._skipped = self.get_scale() < old_scale

        if self.batch_retry_enabled and self.num_retries < self.max_retries and self._skipped:
            if self.should_retry_batch:
                self.num_retries += 1
            self.should_retry_batch = True
        else:
            self.num_retries = 0
            self.should_retry_batch = False
        self._skipped = False
    
    def step(self, optimizer, *args, **kwargs):
        if(self._enabled):
            return super().step(optimizer, *args, **kwargs)
        
        params = [p for pg in optimizer.param_groups for p in pg['params']]
        should_step = True
        for param in params:
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    should_step = False
                    break
        
        if should_step:
            return optimizer.step()
        
        self._skipped = True
        return None
    
    def state_dict(self):
        return {
            'batch_retry_enabled': self.batch_retry_enabled,
            'max_retries': self.max_retries,
            'should_retry_batch': self.should_retry_batch,
            'num_retries': self.num_retries,
            '_skipped': self._skipped,
            **super().state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.batch_retry_enabled = state_dict.pop('batch_retry_enabled')
        self.max_retries = state_dict.pop('max_retries')
        self.should_retry_batch = state_dict.pop('should_retry_batch')
        self.num_retries = state_dict.pop('num_retries')
        self._skipped = state_dict.pop('_skipped')
        return super().load_state_dict(state_dict)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: StatefulDataLoader,
        optimizer: torch.optim.Optimizer,
        config
    ) -> None:
        self.device = config.setup.device
        self.local_rank = config.setup.ddp.local_rank
        self.rank = config.setup.ddp.rank
        self.save_every_batches = config.train.save_every_batches
        self.amp_enabled = config.setup.amp.enabled
        self.amp_dtype = config.setup.amp.dtype
        self.grad_clipping_enabled = config.train.grad_clipping.enabled
        self.max_grad_norm = config.train.grad_clipping.max_norm
        self.max_epochs = config.train.total_epochs

        self.model = model.to(self.device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss = []
        
        # We wrap the model with DDP, giving the GPU IDs where the model is (only in local_rank in this case)
        # This also works for multi-GPU models, but in that case, device_ids and output_device must NOT be set,
        # these should be sent to the proper devices by either the application or by model.forward()
        self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Gradient scaler for AMP (probably not needed if using bfloat16)
        self.grad_manager = GradManager(
            device=self.device,
            enabled=self.amp_enabled and config.setup.grad_manager.scaler.enabled,
            batch_retry_enabled=config.setup.grad_manager.scaler.batch_retry.enabled,
            max_retries=config.setup.grad_manager.scaler.batch_retry.max_retries,
        )
    
        # When using torchrun, we need load and save checkpoint logic because when any of the processes fail, torchrun restarts all of them at the last existing checkpoint
        # Starts from checkpoint if exists
        self.current_epoch = 0
        self.current_batch = 0
        self.current_global_batch = 0
        self.checkpoint_path = config.train.checkpoint_path
        if os.path.exists(self.checkpoint_path):
            print('Loading checkpoint')
            self._load_checkpoint()
        
        get_num_params(self.model.module)

    def _load_checkpoint(self):
        # Maps to the specific device
        # This prevents processes from using others' devices (when set to accelerator:local_rank)
        # TODO I put 'cpu' bc it seems like most people use that, need to check that
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.grad_manager.load_state_dict(checkpoint['grad_manager'])
        self.train_data.load_state_dict(checkpoint['train_data'])
        self.current_epoch = checkpoint['epochs_run']
        self.current_batch = checkpoint['current_batch']
        self.current_global_batch = checkpoint['current_global_batch']

        print(f'Resuming training from checkpoint | Epoch {self.current_epoch + 1}')

    def _save_checkpoint(self):
        # We need .module to access model's parameters since it has been wrapped by DDP
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'grad_manager': self.grad_manager.state_dict(),
            'train_data': self.train_data.state_dict(),
            'epochs_run': self.current_epoch,
            'current_batch': self.current_batch,
            'current_global_batch': self.current_global_batch,
        }

        torch.save(checkpoint, self.checkpoint_path)
        print(f'Saving training checkpoint at {self.checkpoint_path} | Epoch {self.current_epoch + 1}')
        
    def _try_save_new_batch(self):
        self.current_global_batch += 1
        
        # Ensures only saves from first GPU to prevent redundancy
        if self.rank == 0 and self.current_global_batch % self.save_every_batches == 0:
            self._save_checkpoint()
        
    def _should_retry_batch(self):
        return self.grad_manager.should_retry_batch

    # Use this function to run a batch for a generic model
    def _run_batch(self, loss_constructor):
        self.optimizer.zero_grad(set_to_none=True)

        # AMP: Casts operations to mixed precision
        with amp.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp_enabled):
            # output.dtype is bfloat16 because linear layers autocast to bfloat16
            # loss.dtype is float32 because mse_loss layers autocast to float32
            loss = loss_constructor()
            self.loss.append(loss.detach())

        # Exits autocast before backward()
        # Backward passes under autocast are not recommended
        # Backward ops run in the same dtype autocast chose for corresponding forward ops

        # Scales the loss, and calls backward()
        # to create scaled gradients
        self.grad_manager.scale(loss).backward() # Already called in model
        
        # All gradients are scaled in this region up to scaler.step(optimizer), so they need to be unscaled to be used
        # Unscales the gradients of optimizer's assigned params in-place
        self.grad_manager.unscale_(self.optimizer)

        # Gradient clipping
        if self.grad_clipping_enabled:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # TODO

        # Unscales gradients (if not unscaled before) and calls or skips optimizer.step()
        # It skips if there are infs or NaNs in grads
        # Since we called unscale_ before, it will not unscale gradients again
        self.grad_manager.step(self.optimizer)
        
        # Updates the scale for next iteration
        self.grad_manager.update()
        
        return self._should_retry_batch()

    def _run_epoch(self):
        # Setting sampler epoch at beginning of each epoch before creating DataLoader iterator is necessary for shuffling to work in distributed mode across multiple epochs
        # See: https://docs.pytorch.org/docs/stable/data.html
        self.train_data.sampler.set_epoch(self.current_epoch + 1)
        for item in self.train_data:
            scene = preprocess_scene_videos(item, self.device)
            videos, queries, targets, n_frames = scene.sources, scene.queries, scene.targets, scene.n_frames
            self.current_latent_embeds = None # When None, it is initialized to the model's start_latent_embeds
            
            for i in range(0, n_frames, self.model.module.scene_batch_size):
                # TODO save each batch history at checkpoints too
                p_epoch = f'Epoch {self.current_epoch + 1}/{self.max_epochs}'
                p_batch = f'; Batch: {self.current_batch + 1}/{len(self.train_data)}'
                p_frame = f'; Frame: {i + 1}/{n_frames}'
                p_loss = f'; Losses: {[f'{l:.5f}' for l in self.loss]}' if len(self.loss) else ''
                self.loss = []
                print(f'[GPU{self.rank}] {p_epoch}{p_batch}{p_frame}{p_loss}')

                start, end = i, min(n_frames, i + self.model.module.scene_batch_size)
                
                # TODO fix this abomination put in model
                def run_batch1():
                    loss, self.current_latent_embeds = self.model.forward(videos, queries, targets, start, end, self.current_latent_embeds)
                    self.current_latent_embeds = self.current_latent_embeds.detach()
                    # print(self.current_latent_embeds.sum()) # TODO
                    if i != 0:
                        # Adds start_latent_embeds to graph just to prevent the model from complaining bc not all parameters are being optimized
                        loss = loss + (self.model.module.start_latent_embeds).sum() * 0
                    return loss
                # Batch retrying
                while self._run_batch(run_batch1):
                    pass
                    
                self._try_save_new_batch()
                
                if i != 0:
                    def run_batch2():
                        loss, _ = self.model.forward(videos, queries, targets, start, end)
                        return loss
                    # Batch retrying
                    while self._run_batch(run_batch2):
                        pass
                    
                self._try_save_new_batch()
            
            # Saving up memory for next scene
            del scene
            gc.collect()
            torch.cuda.empty_cache()
            
            self.current_batch += 1

    def train(self):
        for self.current_epoch in range(self.current_epoch, self.max_epochs):
            self._run_epoch()
            self.current_batch = 0


def enable_reproducibility(seed, rank):
    # TODO this wont work if restarted

    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    # https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848

    seed = seed + rank
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Impacts performance
    #torch.use_deterministic_algorithms(True)

    # TODO check https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader


def init_ddp(config):
    # Set num threads per process for OpenMP (used by DDP, see https://github.com/pytorch/pytorch/blob/65e6194aeb3269a182cfe2c05c122159da12770f/torch/distributed/run.py#L597-L608)
    # Should be set to num_cpu_threads / num_processes_per_node, that way you have that many threads for each process in the node
    os.environ['OMP_NUM_THREADS'] = str(config.setup.num_threads)

    # Best practice when using DDP with torchrun, since the GPU used for this process will always be the one specified by local_rank
    # This prevents hangs or excessive memory usage on GPU:0
    torch.accelerator.set_device_index(config.setup.ddp.local_rank)

    # Creates process group
    # `backend` is the backend used for inter-GPU communication (will be 'nccl' when device is 'cuda')
    # When using torchrun, we don't need to specify rank and world size since it already handles this for us
    # There are two ways to initialize process group: TCP and shared file-system. See both here: https://docs.pytorch.org/docs/stable/distributed.html#tcp-initialization
    # See backends here: https://docs.pytorch.org/docs/stable/distributed.html#backends
    backend = torch.distributed.get_default_backend_for_device(config.setup.device)
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=config.setup.ddp.timeout))


def prepare_dataloader(dataset: Dataset, config):
    # TODO check if this loader works with ddp. seems to work, but needs to check with multiple gpus
    return StatefulDataLoader(
        dataset,
        batch_size=config.batch_size,
        # Shuffle should be defined in sampler when using DistributedSampler
        shuffle=False,
        # Sampler that sends different batches to different gpus
        sampler=DistributedSampler(dataset, shuffle=config.shuffle),
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=False, # TODO check
        pin_memory=True, # TODO check
        drop_last=False, # TODO check
    )


def prepare_optimizer(model, config):
    # Removing parameters that are not optimized
    params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.AdamW(
        params,
        lr=config.lr,
        betas=config.betas,
        fused=True # TODO Some places report issues so check if this gives errors or nans
    )


def train(config):
    train_dataset = FullDataset(config.train.data.datasets)
    train_data = prepare_dataloader(train_dataset, config.train.data.dataloader)

    model = DVST(config=config.model)

    optimizer = prepare_optimizer(model, config.train.optimizer)

    # We can also do a distributed evaluation by also using distributed sampler in the evaluation data
    # test_data = prepare_dataloader(test_dataset, config.train.data)
    trainer = Trainer(model, train_data, optimizer, config)
    trainer.train()


def main(args):
    config = load_config(args.config_file)

    # Allows tf32 optimization (lower float32 precision with higher speed)
    # This also sets torch.backends.cuda.matmul.allow_tf32, see note in https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision(config.setup.tf32_level)
    torch.backends.cudnn.allow_tf32 = config.setup.tf32_level != 'highest'

    # Enable cuDNN auto-tuner
    # Runs a short benchmark, chooses the best kernel on the first step and uses it in the next steps
    # then the first step is slower but all other steps are faster
    # the problem is that when you have a model that keeps changing at each nth iteration or where that input size changes, it becomes slower since it benchmarks again at every change
    # a rule of thumb would be to run for some time with and without it and check which is faster in the later steps (without considering the first one)
    # This affects reproducibility
    torch.backends.cudnn.benchmark = config.setup.benchmark_kernels

    # Enables reproducibility across runs
    enable_reproducibility(config.setup.seed, config.setup.ddp.rank)

    init_ddp(config)

    try:
        train(config)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--config_file', type=str, help='Path to config file')
    args = parser.parse_args()

    # torchrun already handles setting up env variables and launching processes on the appropriate nodes, so we just call main
    main(args)

    # TODO On using numactl with torchrun:
    # https://github.com/pytorch/pytorch/issues/115305#issuecomment-1845957682
    # https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html#numactl

    # Running single node:
    # torchrun --standalone --nproc-per-node=gpu train.py --config_file=res/config.yaml
    # --standalone: tells it is a single-machine setup
    # --nproc-per-node: num processes per node, can be a number, "gpu" which will create a process per gpu
    # train.py
    #   config_file: res/config.yaml

    # Running multi-node:
    # We run the command in each node, specifying how many nodes in total and the global rank of the current node
    # We also need to set rendezvous arguments to allow them to synchronize and communicate
    # torchrun --nproc-per-node=gpu --nnodes=2 --node-rank=0 --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=172.31.43.139:29603 train.py --config_file=res/config.yaml
    # --nnodes: total num of nodes (can also be in format "min_nodes:max_nodes" where it looks for at least min_nodes and for at most max_nodes, also called elastic launch)
    # --node-rank: current node rank (between 0 and nnodes - 1)
    #   it seems like it does not need to be specified when using SLURM bc it already passes $SLURM_NODEID down
    # --rdzv-id: id for rendezvous protocol, random number
    # --rdzv-backend: backend for rendezvous protocol
    # --rdzv-endpoint: ip and port of any of the participating nodes
    #   the rendezvous backend is hosted here, so its best to choose the one with best bandwidth
    #   it should also be the same for all nodes
    # --max-restarts (optional): number of allowed failures or membership changes (details here: https://docs.pytorch.org/docs/stable/elastic/run.html#membership-changes)

    # E.g. on SLURM enabled cluster, we can find master endpoint by:
    # export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
    # Then we use that like:
    # --rdzv-endpoint=$MASTER_ADDR:29603
