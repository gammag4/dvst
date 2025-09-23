import os
from abc import ABC, abstractmethod
from typing import Generic
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data.distributed import DistributedSampler # Model that takes in data and distributes across GPUs
from torch.nn.parallel import DistributedDataParallel as DDP # DDP wrapper
import torch.distributed as dist
import torch.amp as amp

from src.base.utils import get_model_stats
from src.base.config import Config, TDatasetConfig, TModelConfig, TOptimizerConfig
from src.base.datasets.provider import DatasetProvider
from src.base.model.provider import ModelProvider
from src.base.optimizer.provider import OptimizerProvider, TModel

from .grad_manager import GradManager
from .runner import DistributedRunner


class TrainerResult:
    pass


class DistributedTrainer(DistributedRunner[TDatasetConfig, TModelConfig, TOptimizerConfig, TrainerResult], Generic[TDatasetConfig, TModelConfig, TOptimizerConfig, TModel]):
    def __init__(
        self,
        config: Config[TDatasetConfig, TModelConfig, TOptimizerConfig],
        dataset_provider: DatasetProvider[TDatasetConfig],
        model_provider: ModelProvider[TModelConfig],
        optimizer_provider: OptimizerProvider[TOptimizerConfig, TModel]
    ) -> None:
        super().__init__(config)
        self.dataset_provider = dataset_provider
        self.model_provider = model_provider
        self.optimizer_provider = optimizer_provider
        
        self.device = config.setup.distributed.device
        self.local_rank = config.setup.distributed.local_rank
        self.rank = config.setup.distributed.rank
        self.amp_config = config.setup.amp
        self.grad_scaler_config = config.setup.grad_manager.scaler
        self.grad_clipping_config = config.train.grad_clipping
        self.max_epochs = config.train.total_epochs
    
    def print_current_state(self, extras=''):
        epoch = f'Epoch {self.current_epoch + 1}/{self.max_epochs}'
        batch = f'; Batch: {self.current_batch + 1}/{len(self.train_data)}'
        print(f'[GPU{self.rank}] {epoch}{batch}{extras}')
    
    def _load_checkpoint(self):
        # Maps to the specific device
        # This prevents processes from using others' devices (when set to accelerator:local_rank)
        # TODO I put 'cpu' bc it seems like most people use that, need to check that
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.grad_manager.load_state_dict(checkpoint['grad_manager'])
        self.train_data.load_state_dict(checkpoint['train_data'])
        self.current_epoch = checkpoint['current_epoch']
        self.current_batch = checkpoint['current_batch']
        self.current_global_pass = checkpoint['current_global_pass']
        
        print(f'Resuming training from checkpoint | Epoch {self.current_epoch + 1}')
    
    def _save_checkpoint(self):
        # We need .module to access model's parameters since it has been wrapped by DDP
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'grad_manager': self.grad_manager.state_dict(),
            'train_data': self.train_data.state_dict(),
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'current_global_pass': self.current_global_pass,
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        print(f'Saving training checkpoint at {self.checkpoint_path} | Epoch {self.current_epoch + 1}')
    
    def _load_extras(self):
        pass
    
    def _save_extras(self):
        pass
    
    def _create_dataloader(self, dataset: Dataset):
        config = self.config.train.data.dataloader
        
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
            pin_memory=config.pin_memory, # TODO check
            drop_last=False, # TODO check
        )
    
    def _try_save_new_pass(self):
        self.current_global_pass += 1
        
        # Ensures only saves from first GPU to prevent redundancy
        if self.rank == 0 and self.current_global_pass % self.config.train.save_every_passes == 0:
            self._save_checkpoint()
    
    def _should_retry_pass(self):
        return self.grad_manager.should_retry_batch
    
    # This method is automatically called by self._run_pass() and should not be called by the user
    # It receives the same *args sent to self._run_pass() by self._run_dataset_batch() and should do one forward pass, returning the loss of that pass
    @abstractmethod
    def _run_forward(self, *args):
        pass
    
    # Use this function to run a batch for a generic model
    def _run_pass_once(self, *args):
        self.optimizer.zero_grad(set_to_none=True)
        
        # AMP: Casts operations to mixed precision
        with amp.autocast(device_type=self.device, dtype=self.amp_config.dtype, enabled=self.amp_config.enabled):
            # output.dtype is bfloat16 because linear layers autocast to bfloat16
            # loss.dtype is float32 because mse_loss layers autocast to float32
            loss = self._run_forward(*args)
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
        if self.grad_clipping_config.enabled:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping_config.max_norm) # TODO
        
        # Unscales gradients (if not unscaled before) and calls or skips optimizer.step()
        # It skips if there are infs or NaNs in grads
        # Since we called unscale_ before, it will not unscale gradients again
        self.grad_manager.step(self.optimizer)
        
        # Updates the scale for next iteration
        self.grad_manager.update()
        
        should_retry = self._should_retry_pass()
        if not should_retry:
            self._try_save_new_pass()

        return should_retry
    
    # Use this method to run one forward/backward pass for a generic model
    def _run_pass(self, *args):
        # Batch retrying
        while self._run_pass_once(*args):
            pass
    
    # This method receives one batch directly from the dataset and should train the model with it, should not be called by the used
    # It needs to call self._run_pass() for each training pass it needs to run (forward and backward), where *args will be the args it will pass down to self._run_forward() when calling it
    @abstractmethod
    def _run_dataset_batch(self, batch):
        pass
    
    def _run_epoch(self):
        # Setting sampler epoch at beginning of each epoch before creating DataLoader iterator is necessary for shuffling to work in distributed mode across multiple epochs
        # See: https://docs.pytorch.org/docs/stable/data.html
        self.train_data.sampler.set_epoch(self.current_epoch)
        for batch in self.train_data:
            self.print_current_state()
            self._run_dataset_batch(batch)
            self.current_batch += 1
    
    def _train(self):
        for self.current_epoch in range(self.current_epoch, self.max_epochs):
            self._run_epoch()
            self.current_batch = 0
        
        return TrainerResult()
    
    async def _run(self):
        dataset_config = self.config.train.data.dataset
        
        await self.dataset_provider.download_dataset(dataset_config)
        
        train_data = self.dataset_provider.create_train_dataset(dataset_config)
        self.train_data = self._create_dataloader(train_data)
        
        val_data = self.dataset_provider.create_val_dataset(dataset_config)
        self.val_data = self._create_dataloader(val_data) # TODO
        
        model = self.model_provider.create_model(self.config.model)
        self.optimizer = self.optimizer_provider.create_optimizer(self.config.train.optimizer, model)
        
        # We wrap the model with DDP, giving the GPU IDs where the model is (only in local_rank in this case)
        # This also works for multi-GPU models, but in that case, device_ids and output_device must NOT be set,
        # these should be sent to the proper devices by either the application or by model.forward()
        self.model = DDP(model.to(self.device), device_ids=[self.local_rank])
        
        self.loss = []
        
        # Gradient scaler for AMP (probably not needed if using bfloat16)
        self.grad_manager = GradManager(
            device=self.device,
            enabled=self.amp_config.enabled and self.grad_scaler_config.enabled,
            batch_retry_enabled=self.grad_scaler_config.batch_retry.enabled,
            max_retries=self.grad_scaler_config.batch_retry.max_retries,
        )
        
        # When using torchrun, we need load and save checkpoint logic because when any of the processes fail, torchrun restarts all of them at the last existing checkpoint
        # Starts from checkpoint if exists
        self.current_epoch = 0
        self.current_batch = 0
        self.current_global_pass = 0
        self.checkpoints_folder_path = self.config.train.checkpoints_folder_path
        self.checkpoint_path = os.path.join(self.checkpoints_folder_path, 'checkpoint.pt')
        os.makedirs(self.checkpoints_folder_path, exist_ok=True)
        if os.path.exists(self.checkpoint_path):
            print('Loading checkpoint')
            self._load_checkpoint()
        
        get_model_stats(self.model.module)
        
        return self._train()
