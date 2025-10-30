import os
import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, cast
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data.distributed import DistributedSampler # Model that takes in data and distributes across GPUs
from torch.nn.parallel import DistributedDataParallel as DDP # DDP wrapper
import torch.distributed as dist
import torch.amp as amp

from src.base.utils import print_model_stats
from src.base.config import Config, TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig
from src.base.datasets.provider import DatasetProvider
from src.base.model.provider import ModelProvider
from src.base.optimizer.provider import OptimizerProvider, TModel
from src.base.loss.provider import LossProvider

from .grad_manager import GradManager
from .logging import Stateful
from .logging.providers import LogProvider
from .timer import Timer
from .runner import DistributedRunner


@dataclass
class TrainerResult:
    score: float


class DistributedTrainer(DistributedRunner[TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig, TrainerResult], Stateful, Generic[TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig, TModel]):
    def __init__(
        self,
        config: Config[TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig],
        dataset_provider: DatasetProvider[TDatasetConfig],
        model_provider: ModelProvider[TModelConfig],
        optimizer_provider: OptimizerProvider[TOptimizerConfig, TModel],
        loss_provider: LossProvider[TLossConfig],
        log_provider: LogProvider
    ) -> None:
        super().__init__(config)
        self.dataset_provider = dataset_provider
        self.model_provider = model_provider
        self.optimizer_provider = optimizer_provider
        self.loss_provider = loss_provider
        self.log_provider = log_provider
        
        self.device = config.setup.distributed.device
        self.local_rank = config.setup.distributed.local_rank
        self.rank = config.setup.distributed.rank
        self.amp_config = config.setup.amp
        self.grad_scaler_config = config.setup.grad_manager.scaler
        self.grad_clipping_config = config.train.grad_clipping
        self.max_epochs = config.train.total_epochs
    
    @property
    def base_model(self):
        return cast(TModel, self.model.module)
    
    @property
    @abstractmethod
    def n_train_steps(self):
        pass
    
    @property
    @abstractmethod
    def n_val_steps(self):
        pass
    
    @property
    def total_steps(self):
        # TODO add val dataset
        return self.n_train_steps
    
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
    
    def load_default_state(self):
        self.current_epoch = 0
        self.current_global_pass = 0
    
    def state_dict(self):
        state_dict = {
            'loss_scheduler': self.loss_scheduler.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'grad_manager': self.grad_manager.state_dict(),
            'logger': self.logger.state_dict(),
            'timer': self.timer.state_dict(),
            'current_epoch': self.current_epoch,
            'current_global_pass': self.current_global_pass,
        }
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.loss_scheduler.load_state_dict(state_dict['loss_scheduler'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.grad_manager.load_state_dict(state_dict['grad_manager'])
        self.logger.load_state_dict(state_dict['logger'])
        self.timer.load_state_dict(state_dict['timer'])
        self.current_epoch = state_dict['current_epoch']
        self.current_global_pass = state_dict['current_global_pass']
    
    def _try_load_checkpoint(self):
        config = self.config.train.checkpoints
        
        os.makedirs(config.folder_path, exist_ok=True)
        model_checkpoints = [i for i in os.listdir(config.folder_path) if i.split('.')[0].isdigit()]
        if len(model_checkpoints) > 0:
            last_checkpoint = max(model_checkpoints, key=lambda x: int(x.split('.')[0]))
            checkpoint_path = os.path.join(config.folder_path, last_checkpoint)
            
            # Maps to the specific device
            # This prevents processes from using others' devices (when set to accelerator:local_rank)
            # TODO I put 'cpu' bc it seems like most people use that, need to check that
            # TODO weights_only=False only for trusted
            self.base_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=config.weights_only))
            self.logger.message(f'Resumed training with model from {checkpoint_path}')
        
        train_checkpoint = os.path.join(config.folder_path, 'train_data.pt')
        if os.path.isfile(train_checkpoint):
            self.load_state_dict(torch.load(train_checkpoint, map_location='cpu', weights_only=config.weights_only))
            self.logger.message(f'Resumed training with training data from {train_checkpoint}')
        else:
            self.load_default_state()
    
    def _try_save_checkpoint(self):
        config = self.config.train.checkpoints
        
        # Ensures only saves from first GPU to prevent redundancy
        if self.rank == 0 and self.current_global_pass % self.config.train.save_every_passes == 0:
            torch.accelerator.synchronize(self.device)
            
            checkpoint_path = os.path.join(config.folder_path, f'{self.logger.iteration}.pt')
            torch.save(self.base_model.state_dict(), checkpoint_path)
            self.logger.message(f'Saved trained model at {checkpoint_path}')
            
            train_checkpoint = os.path.join(config.folder_path, 'train_data.pt')
            train_checkpoint_temp = os.path.join(config.folder_path, 'train_data_temp.pt')
            torch.save(self.state_dict(), train_checkpoint_temp)
            os.replace(train_checkpoint_temp, train_checkpoint)
            self.logger.message(f'Saved training data at {train_checkpoint}')
    
    def _should_retry_pass(self):
        return self.grad_manager.should_retry_batch
    
    # This method is automatically called by self._run_pass() and should not be called by the user
    # It receives the same *args sent to self._run_pass() by self._run_epoch() and should do one forward pass, returning the loss of that pass
    # This method also has to call ddp model forward, can't call model.module functions directly
    # See https://discuss.pytorch.org/t/is-it-ok-to-use-methods-other-than-forward-in-ddp/176509
    @abstractmethod
    def _run_forward(self, *args):
        pass
    
    def _val(self):
        pass # TODO
    
    # Use this function to run a batch for a generic model
    def _run_pass_once(self, *args):
        self.optimizer.zero_grad(set_to_none=True)
        
        # AMP: Casts operations to mixed precision
        with amp.autocast(device_type=self.device, dtype=self.amp_config.dtype, enabled=self.amp_config.enabled):
            # output.dtype is bfloat16 because linear layers autocast to bfloat16
            # loss.dtype is float32 because mse_loss layers autocast to float32
            loss = self._run_forward(*args)
        
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
        
        self.logger.log({'loss': loss.detach().to('cpu', non_blocking=True)})
    
    # This method is run after each pass to update stuff
    def _step(self):
        self.timer.update()

        self.logger.log({
            'avg_delta_time': self.timer.avg_delta,
            'eta': str(datetime.timedelta(seconds=self.timer.eta))
        })
        
        if self.loss_scheduler is not None:
            self.loss_scheduler.step()
        
        if self.lr_scheduler is not None and not self.grad_manager.skipped_last:
            self.lr_scheduler.step()
    
    # Use this method to run one forward/backward pass for a generic model
    def _run_pass(self, *args):
        # Batch retrying
        while True:
            self._run_pass_once(*args)
            if not self._should_retry_pass():
                break
        
        self._step()
        
        self.current_global_pass += 1
        
        if self.rank == 0 and self.current_global_pass % self.config.train.log_every_passes == 0:
            torch.accelerator.synchronize(self.device)
            self.logger.display_current()
        self.logger.update()
        
        self._try_save_checkpoint()
    
    # This method is called in each epoch with self.current_epoch as its epoch number
    # It needs to call self._run_pass() for each training pass it needs to run (forward and backward) for each batch, where *args will be the args it will pass down to self._run_forward() when calling it
    @abstractmethod
    def _run_epoch(self):
        pass
    
    def _train(self):
        self.logger.log({'gpu': self.rank})
        
        for self.current_epoch in range(self.current_epoch, self.max_epochs):
            self.logger.log({'epoch': self.current_epoch})
            
            self._run_epoch()
        
        return TrainerResult(
            score=0.0 # TODO
        )
    
    async def _run(self):
        loss, self.loss_scheduler = self.loss_provider.create_loss(self.config.train.loss, self.total_steps)
        model = self.model_provider.create_model(self.config.model, loss)
        self.optimizer, self.lr_scheduler = self.optimizer_provider.create_optimizer(self.config.train.optimizer, model, self.total_steps)
        
        # We wrap the model with DDP, giving the GPU IDs where the model is (only in local_rank in this case)
        # This also works for multi-GPU models, but in that case, device_ids and output_device must NOT be set,
        # these should be sent to the proper devices by either the application or by model.forward()
        self.model = DDP(model.to(self.device, non_blocking=True), device_ids=[self.local_rank])
        
        # Gradient scaler for AMP (probably not needed if using bfloat16)
        self.grad_manager = GradManager(
            device=self.device,
            enabled=self.amp_config.enabled and self.grad_scaler_config.enabled,
            batch_retry_enabled=self.grad_scaler_config.batch_retry.enabled,
            max_retries=self.grad_scaler_config.batch_retry.max_retries,
        )
        
        self.logger = self.log_provider.create_logger()
        
        self.timer = Timer(self.total_steps)
        
        # When using torchrun, we need load and save checkpoint logic because when any of the processes fail, torchrun restarts all of them at the last existing checkpoint
        # Starts from checkpoint if exists
        self._try_load_checkpoint()
        
        print_model_stats(self.model)
        
        return self._train()


class DefaultDistributedTrainer(DistributedTrainer[TDatasetConfig, TModelConfig, TOptimizerConfig, TLossConfig, TModel]):
    @property
    def n_train_steps(self):
        return len(self.train_data) * self.max_epochs
    
    @property
    def n_val_steps(self):
        return len(self.val_data) * self.max_epochs
    
    def _run_forward(self, *args):
        x, y = args
        res = self.model(x)
        return self.loss(res, y)
    
    def load_default_state(self):
        super().load_default_state()
        
        self.current_batch = 0
    
    # Override to save more stuff in checkpoints
    def state_dict(self):
        state_dict = super().state_dict()
        
        state_dict['current_batch'] = self.current_batch
        state_dict['train_data'] = self.train_data.state_dict()
        state_dict['val_data'] = self.val_data.state_dict()
        
        return state_dict
    
    # Override to save more stuff in checkpoints
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        
        self.train_data.load_state_dict(state_dict['train_data'])
        self.val_data.load_state_dict(state_dict['val_data'])
        self.current_batch = state_dict['current_batch']
    
    def _run_epoch(self):
        # Setting sampler epoch at beginning of each epoch before creating DataLoader iterator is necessary for shuffling to work in distributed mode across multiple epochs
        # See: https://docs.pytorch.org/docs/stable/data.html
        self.train_data.sampler.set_epoch(self.current_epoch)
        
        for batch in self.train_data:
            self.logger.log({'batch': self.current_batch})
            
            self._run_pass(batch)
            self.current_batch += 1
        
        self.current_batch = 0
    
    async def _run(self):
        dataset_config = self.config.train.data.dataset
        
        await self.dataset_provider.download_dataset(dataset_config)
        
        train_data = self.dataset_provider.create_train_dataset(dataset_config)
        self.train_data = self._create_dataloader(train_data)
        
        val_data = self.dataset_provider.create_val_dataset(dataset_config)
        self.val_data = self._create_dataloader(val_data)
        
        return await super()._run()
