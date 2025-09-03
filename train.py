import os
import random
import datetime

import numpy as np
import torch

import torch.distributed as dist

from src.config import load_config


def enable_reproducibility(config):
    # TODO this wont work if restarted
    
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    # https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848
    
    seed = config.seed + config.distributed.rank
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Impacts performance
    #torch.use_deterministic_algorithms(True)
    
    # TODO check https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader


def setup_optimizations(config):
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


def init_distributed(config):
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
    backend = torch.distributed.get_default_backend_for_device(config.device)
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=config.distributed.timeout))


def main(args):
    config = load_config(args.config_file)
    
    enable_reproducibility(config.setup)
    setup_optimizations(config.setup)
    
    init_distributed(config.setup)
    
    try:
        trainer = config.setup.trainer_constructor(config)
        trainer.run()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--config_file', type=str, help='Path to config file')
    args = parser.parse_args()
    
    # torchrun already handles setting up env variables and launching processes on the appropriate nodes
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
