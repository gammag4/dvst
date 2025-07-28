import os
import random
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp # Wrapper around python's native multiprocessing
from torch.utils.data.distributed import DistributedSampler # Model that takes in data and distributes across GPUs
from torch.nn.parallel import DistributedDataParallel as DDP # DDP wrapper
import torch.distributed as dist
import torch.amp as amp

from src.config import load_config
from datautils import MyTrainDataset


class GradScaler(amp.GradScaler):
    def __init__(self, device, enabled, batch_replay_enabled, max_replays):
        super().__init__(device=device, enabled=enabled)
        self.batch_replay_enabled = batch_replay_enabled
        self.max_replays = max_replays
        self.should_replay_batch = False
        self.num_replays = 0
    
    def update(self):
        old_scale = self.get_scale()
        super().update()

        # Checks whether current batch has been skipped due to inf/nan grads
        # If the scale is smaller than before, it means that it updated its scale because of inf/nan grads
        if self.batch_replay_enabled and self.get_scale() < old_scale and self.num_replays < self.max_replays:
            if self.should_replay_batch:
                self.num_replays += 1
            self.should_replay_batch = True
        else:
            self.num_replays = 0
            self.should_replay_batch = False
    
    def state_dict(self):
        return {
            'batch_replay_enabled': self.batch_replay_enabled,
            'max_replays': self.max_replays,
            'should_replay_batch': self.should_replay_batch,
            'num_replays': self.num_replays,
            **super().state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.batch_replay_enabled = state_dict.pop('batch_replay_enabled')
        self.max_replays = state_dict.pop('max_replays')
        self.should_replay_batch = state_dict.pop('should_replay_batch')
        self.num_replays = state_dict.pop('num_replays')
        return super().load_state_dict(state_dict)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        config
    ) -> None:
        self.device = config.setup.device
        self.local_rank = config.setup.ddp.local_rank
        self.rank = config.setup.ddp.rank
        self.save_every = config.train.save_every
        self.amp_enabled = config.setup.amp.enabled
        self.amp_dtype = config.setup.amp.dtype
        self.grad_clipping_enabled = config.train.grad_clipping.enabled
        self.max_grad_norm = config.train.grad_clipping.max_norm

        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer

        # When using torchrun, we need load and save checkpoint logic because when any of the processes fail, torchrun restarts all of them at the last existing checkpoint
        # Starts from checkpoint if exists
        self.epochs_run = 0
        self.checkpoint_path = config.train.checkpoint_path
        if os.path.exists(self.checkpoint_path):
            print('Loading checkpoint')
            self._load_checkpoint()

        # We wrap the model with DDP, giving the GPU IDs where the model is (only in local_rank in this case)
        # This also works for multi-GPU models, but in that case, device_ids and output_device must NOT be set,
        # these should be sent to the proper devices by either the application or by model.forward()
        self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Gradient scaler for AMP
        self.scaler = GradScaler(
            device=self.device,
            enabled=self.amp_enabled and config.setup.amp.scaler.enabled,
            batch_replay_enabled=config.setup.amp.scaler.batch_replay.enabled,
            max_replays=config.setup.amp.scaler.batch_replay.max_replays,
        )

    def _load_checkpoint(self):
        # Maps to the specific device
        # This prevents processes from using others' devices (when set to accelerator:local_rank)
        # TODO I put 'cpu' bc it seems like most people use that, need to check that
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.epochs_run = checkpoint['epochs_run']

        print(f'Resuming training from checkpoint at Epoch {self.epochs_run}')

    def _save_checkpoint(self, epoch):
        # We need .module to access model's parameters since it has been wrapped by DDP
        checkpoint = {
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epochs_run': epoch,
        }

        torch.save(checkpoint, self.checkpoint_path)
        print(f'Epoch {epoch} | Training checkpoint saved at {self.checkpoint_path}')
        
    def _should_replay_batch(self):
        return self.scaler.should_replay_batch()

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad(set_to_none=True)

        # AMP
        with amp.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp_enabled):
            output = self.model(source)
            loss = F.cross_entropy(output, targets)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        if self.grad_clipping_enabled:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # TODO
        
        # Skips steps with inf/nan gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return self._should_replay_batch()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f'[GPU{self.rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}')

        # Setting sampler epoch at beginning of each epoch before creating DataLoader iterator is necessary for shuffling to work in distributed mode across multiple epochs
        # See: https://docs.pytorch.org/docs/stable/data.html
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            
            # Batch replaying
            while self._run_batch(source, targets):
                pass

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # Ensures only saves for first GPU to prevent redundancy
            if self.rank == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():#
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        # Should be false bc using sampler
        shuffle=False,
        # Sampler that sends different batches to different gpus
        sampler=DistributedSampler(dataset)
    )


def enable_reproducibility(seed, rank):
    # TODO this wont work if restarted

    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    # https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848

    seed = seed + rank
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Impacts performance
    #torch.use_deterministic_algorithms(True)

    # TODO check https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader


def init_ddp(config):
    # No need for this since torchrun already specifies master addr and port
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'

    # Set num threads per process for OpenMP (used by DDP, see https://github.com/pytorch/pytorch/blob/65e6194aeb3269a182cfe2c05c122159da12770f/torch/distributed/run.py#L597-L608)
    # Should be set to num_cpu_threads / num_processes_per_node, that way you have that many threads for each process in the node
    num_cpus = os.cpu_count()
    os.environ['OMP_NUM_THREADS'] = num_cpus // config.setup.ddp.local_world_size + (1 if config.setup.ddp.local_rank > num_cpus % config.setup.ddp.local_world_size else 0)

    # Best practice when using DDP with torchrun, since the GPU used for this process will always be the one specified by local_rank
    # This prevents hangs or excessive memory usage on GPU:0
    torch.accelerator.set_device_index(config.setup.ddp.local_rank)

    # Creates process group
    # `backend` is the backend used for inter-GPU communication (will be 'nccl' when device is 'cuda')
    # When using torchrun, we don't need to specify rank and world size since it already handles this for us
    # There are two ways to initialize process group: TCP and shared file-system. See both here: https://docs.pytorch.org/docs/stable/distributed.html#tcp-initialization
    # See backends here: https://docs.pytorch.org/docs/stable/distributed.html#backends
    backend = torch.distributed.get_default_backend_for_device(config.setup.device)
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=config.setup.ddp.timeout)) #, rank=rank, world_size=world_size)


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

    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, config.train.batch_size)
    # We can also do a distributed evaluation by also using distributed sampler in the evaluation data
    # test_data = prepare_dataloader(test_dataset, config.train.batch_size)
    trainer = Trainer(model, train_data, optimizer, config)
    trainer.train(config.train.total_epochs)

    # Destroys process group
    dist.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--config_file', type=str, help='Path to config file')
    args = parser.parse_args()

    # No need for this when using torchrun
    # # no "rank" arg since mp.spawn already passes "rank" down as first argument
    # # we set nprocs=world_size to create as many processes as the number of GPUs (1 process per GPU)
    # #world_size = torch.accelerator.device_count()
    # More details here: https://docs.pytorch.org/docs/stable/distributed.html#spawn-utility
    # #mp.spawn(main, args=(config), nprocs=world_size)

    # torchrun already handles setting up env variables and launching processes on the appropriate nodes, so we just call main
    main(args)

    # TODO On using numactl with torchrun:
    # https://github.com/pytorch/pytorch/issues/115305#issuecomment-1845957682
    # https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html#numactl

    # Running single node:
    # torchrun --standalone --nproc-per-node=gpu script.py 50 10
    # --standalone: tells it is a single-machine setup
    # --nproc-per-node: num processes per node, can be a number, "gpu" which will create a process per gpu
    # script.py
    #   total_epochs: 50
    #   save_every: 10

    # Running multi-node:
    # We run the command in each node, specifying how many nodes in total and the global rank of the current node
    # We also need to set rendezvous arguments to allow them to synchronize and communicate
    # torchrun --nproc-per-node=gpu --nnodes=2 --node-rank=0 --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=172.31.43.139:29603 script.py 50 10
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
