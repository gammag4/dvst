import asyncio

from src.base.providers import ConfigProvider, DatasetProvider, ModelProvider, OptimizerProvider, RunProvider

from src.dvst.providers import DVSTConfigProvider, DVSTDatasetProvider, DVSTModelProvider, DVSTOptimizerProvider, DVSTRunProvider

async def main(
    config_provider: ConfigProvider,
    dataset_provider: DatasetProvider,
    model_provider: ModelProvider,
    optimizer_provider: OptimizerProvider,
    run_provider: RunProvider
):
    config = config_provider.get_default_config()
    trainer = run_provider.create_trainer(config)
    await trainer.run()


if __name__ == '__main__':
    # TODO maybe use pydantic
    config_provider = DVSTConfigProvider()
    dataset_provider = DVSTDatasetProvider()
    model_provider = DVSTModelProvider()
    optimizer_provider = DVSTOptimizerProvider()
    run_provider = DVSTRunProvider(dataset_provider, model_provider, optimizer_provider)
    asyncio.run(main(
        config_provider=config_provider,
        dataset_provider=dataset_provider,
        model_provider=model_provider,
        optimizer_provider=optimizer_provider,
        run_provider=run_provider
    ))
    
    # torchrun already handles setting up env variables and launching processes on the appropriate nodes
 
    # TODO On using numactl with torchrun:
    # https://github.com/pytorch/pytorch/issues/115305#issuecomment-1845957682
    # https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html#numactl
    
    # Running single node:
    # torchrun --standalone --nproc-per-node=gpu train.py
    # --standalone: tells it is a single-machine setup
    # --nproc-per-node: num processes per node, can be a number, "gpu" which will create a process per gpu
    # train.py
    #   config_file: res/config.yaml
    
    # Running multi-node:
    # We run the command in each node, specifying how many nodes in total and the global rank of the current node
    # We also need to set rendezvous arguments to allow them to synchronize and communicate
    # torchrun --nproc-per-node=gpu --nnodes=2 --node-rank=0 --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=172.31.43.139:29603 train.py
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
