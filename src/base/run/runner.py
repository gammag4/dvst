import os
import random
import datetime
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import numpy as np
import torch
import torch.distributed as dist

from src.base.config import Config, TDatasetConfig, TModelConfig, TOptimizerConfig


TRunnerResult = TypeVar('TRunnerResult')


class DistributedRunner(ABC, Generic[TDatasetConfig, TModelConfig, TOptimizerConfig, TRunnerResult]):
    def __init__(self, config: Config[TDatasetConfig, TModelConfig, TOptimizerConfig]):
        self.config = config
    
    def _enable_reproducibility(self):
        config = self.config.setup
        
        # TODO this wont work if restarted
        
        # https://docs.pytorch.org/docs/stable/notes/randomness.html
        # https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848
        
        seed = config.seed + config.distributed.rank
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Impacts performance
        torch.use_deterministic_algorithms(config.use_deterministic_algorithms)
        
        # TODO check https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader
    
    def _setup_optimizations(self):
        config = self.config.setup
        
        # Allows tf32 optimization (lower float32 precision with higher speed)
        # This also sets torch.backends.cuda.matmul.allow_tf32, see note in https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
        torch.set_float32_matmul_precision(config.tf32_level)
        torch.backends.cudnn.allow_tf32 = config.tf32_level != 'highest'
        
        # Enable cuDNN auto-tuner
        # Runs a short benchmark, chooses the best kernel on the first step and uses it in the next steps
        # then the first step is slower but all other steps are faster
        # the problem is that when you have a model that keeps changing at each nth iteration or where that input size changes, it becomes slower since it benchmarks again at every change
        # a rule of thumb would be to run for some time with and without it and check which is faster in the later steps (without considering the first one)
        # This affects reproducibility
        torch.backends.cudnn.benchmark = config.benchmark_kernels
    
    def _init_distributed(self):
        config = self.config.setup
        
        # Set num threads per process for OpenMP (used by DDP, see https://github.com/pytorch/pytorch/blob/65e6194aeb3269a182cfe2c05c122159da12770f/torch/distributed/run.py#L597-L608)
        # Should be set to num_cpu_threads / num_processes_per_node, that way you have that many threads for each process in the node
        os.environ['OMP_NUM_THREADS'] = str(config.distributed.num_threads)
        
        # Best practice when using DDP with torchrun, since the GPU used for this process will always be the one specified by local_rank
        # This prevents hangs or excessive memory usage on GPU:0
        torch.accelerator.set_device_index(config.distributed.local_rank)
        
        # Creates process group
        # `backend` is the backend used for inter-GPU communication (will be 'nccl' when device is 'cuda')
        # When using torchrun, we don't need to specify rank and world size since it already handles this for us
        # There are two ways to initialize process group: TCP and shared file-system. See both here: https://docs.pytorch.org/docs/stable/distributed.html#tcp-initialization
        # See backends here: https://docs.pytorch.org/docs/stable/distributed.html#backends
        backend = torch.distributed.get_default_backend_for_device(config.distributed.device)
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=config.distributed.timeout))
    
    def _end_distributed(self):
        dist.destroy_process_group()
    
    @abstractmethod
    async def _run(self) -> TRunnerResult:
        pass
    
    async def run(self) -> TRunnerResult:
        self._enable_reproducibility()
        self._setup_optimizations()
        
        self._init_distributed()
        
        try:
            return await self._run()
        finally:
            self._end_distributed()
