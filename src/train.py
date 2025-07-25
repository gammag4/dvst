import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp # Wrapper around python's native multiprocessing
from torch.utils.data.distributed import DistributedSampler # Model that takes in data and distributes across GPUs
from torch.nn.parallel import DistributedDataParallel as DDP # DDP wrapper
from torch.distributed import init_process_group, destroy_process_group # Initialize and destroy the distributed process group
import os


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        local_rank: int,
        global_rank: int
    ) -> None:
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

        # When using torchrun, we need load and save checkpoint logic because when any of the processes fail, torchrun restarts all of them at the last existing snapshot
        # Starts from snapshot if exists
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()

        # We wrap the model with DDP, giving the GPU IDs where the model is (only in local_rank in this case)
        # This also works for multi-GPU models, but in that case, device_ids and output_device must NOT be set,
        # these should be sent to the proper devices by either the application or by model.forward()
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self):
        # Maps to the specific cuda device
        # This prevents processes from using others' devices
        snapshot = torch.load(self.snapshot_path, map_location=f"cuda:{self.local_rank}")
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        # We need .module to access model's parameters since it has been wrapped by DDP
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {self.snapshot_path}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        # Setting sampler epoch at beginning of each epoch before creating DataLoader iterator is necessary for shuffling to work in distributed mode across multiple epochs
        # See: https://docs.pytorch.org/docs/stable/data.html
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # Ensures only saves for first GPU to prevent redundancy
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
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


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str):
    # Torchrun already sets local and global ranks as environment variables
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    # No need for this since torchrun already specifies master addr and port
    #os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "12355"

    # Best practice when using DDP with torchrun, since the GPU used for this process will always be the one specified by local_rank
    # This prevents hangs or excessive memory usage on GPU:0
    torch.cuda.set_device(local_rank)
    # Creates process group
    # NCCL is a NVIDIA backend used for inter-GPU communication
    # When using torchrun, we don't need to specify rank and world size since it already handles this for us
    # There are two ways to initialize process group: TCP and shared file-system. See both here: https://docs.pytorch.org/docs/stable/distributed.html#tcp-initialization
    # See backends here: https://docs.pytorch.org/docs/stable/distributed.html#backends
    init_process_group(backend="nccl") #, rank=rank, world_size=world_size)

    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    # We can also do a distributed evaluation by also using distributed sampler in the evaluation data
    # test_data = prepare_dataloader(test_dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path, local_rank, global_rank)
    trainer.train(total_epochs)

    # Destroys process group
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    # No need for this when using torchrun
    # # no "rank" arg since mp.spawn already passes "rank" down
    # # we set nprocs=world_size to create as many processes as the number of GPUs (1 process per GPU)
    # #world_size = torch.cuda.device_count()
    # More details here: https://docs.pytorch.org/docs/stable/distributed.html#spawn-utility
    # #mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)

    # torchrun already handles setting up env variables and launching processes on the appropriate nodes, so we just call main
    main(args.save_every, args.total_epochs, args.batch_size, "snapshot.pt")

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
