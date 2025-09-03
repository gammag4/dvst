import os
import gc
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torch.utils.data.distributed import DistributedSampler # Model that takes in data and distributes across GPUs
from torch.nn.parallel import DistributedDataParallel as DDP # DDP wrapper
import torch.distributed as dist
import torch.amp as amp

from src.datasets.full_dataset import FullDataset
from src.utils import get_num_params


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


class DistributedTrainer(ABC):
    def __init__(
        self,
        config
    ) -> None:
        self.device = config.setup.device
        self.local_rank = config.setup.distributed.local_rank
        self.rank = config.setup.distributed.rank
        self.save_every_passes = config.train.save_every_passes
        self.amp_enabled = config.setup.amp.enabled
        self.amp_dtype = config.setup.amp.dtype
        self.grad_clipping_enabled = config.train.grad_clipping.enabled
        self.max_grad_norm = config.train.grad_clipping.max_norm
        self.max_epochs = config.train.total_epochs
        
        #TODO fix abomination decouple
        
        self.config = config
        
        self._download_datasets()
        
        dataset = self._create_dataset()
        train_data = self._create_dataloader(dataset)
        # train_dataloader = self._create_dataloader(dataset)
        # We can also do a distributed evaluation by also using distributed sampler in the evaluation data
        # test_dataloader = self._create_dataloader(test_dataset)
        
        model = self._create_model()
        
        optimizer = self._create_optimizer(model)
        
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
        self.current_global_pass = 0
        self.checkpoint_folder_path = config.train.checkpoint_folder_path
        self.checkpoint_path = os.path.join(self.checkpoint_folder_path, 'checkpoint.pt')
        os.makedirs(self.checkpoint_folder_path, exist_ok=True)
        if os.path.exists(self.checkpoint_path):
            print('Loading checkpoint')
            self._load_checkpoint()
        
        get_num_params(self.model.module)
    
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
    
    def _download_datasets(self):
        config = self.config
        
        # TODO fix downloaders code, get from cloud and also only download in one process per node
        data_downloaders = config.train.data.downloaders
        for downloader in data_downloaders:
            downloader.download()
    
    @abstractmethod
    def _create_dataset(self) -> Dataset:
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
            pin_memory=True, # TODO check
            drop_last=False, # TODO check
        )
    
    def _create_model(self):
        config = self.config.model
        
        return config.constructor(config=config)
    
    @abstractmethod
    def _create_optimizer(self, model):
        pass
    
    def _try_save_new_pass(self):
        self.current_global_pass += 1
        
        # Ensures only saves from first GPU to prevent redundancy
        if self.rank == 0 and self.current_global_pass % self.save_every_passes == 0:
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
        with amp.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp_enabled):
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
        if self.grad_clipping_enabled:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # TODO
        
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
    
    def run(self):
        for self.current_epoch in range(self.current_epoch, self.max_epochs):
            self._run_epoch()
            self.current_batch = 0


class DVSTTrainer(DistributedTrainer):
    def _create_dataset(self):
        config = self.config.train.data.datasets
        
        # TODO split dataset in a way where each distributed process receives roughly the same amount of batches
        return FullDataset(config)
    
    def _create_optimizer(self, model):
        config = self.config.train.optimizer
        
        # Removing parameters that are not optimized
        params = [p for p in model.parameters() if p.requires_grad]
        
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=config.betas,
            fused=True # TODO Some places report issues so check if this gives errors or nans
        )
    
    def _run_forward(self, *args):
        scene_batch, = args
        loss, _ = self.model.forward(scene_batch)
        return loss
    
    def _run_dataset_batch(self, batch):
        scene = batch.load_scene(self.model.module.scene_batch_size, self.device)
        
        for i, scene_batch in enumerate(scene):
            # TODO save each batch history at checkpoints too put to separate function
            #   in same function also save latent_embeds data sum mean var add it explicitly in function as extra args
            #   add everything and log it to a file
            #   then visualize everything with a notebook
            p_frame = f'; Frame: {i * scene.batch_size + 1}/{scene.n_frames}'
            p_loss = f'; Losses: {[f'{l:.5f}' for l in self.loss]}' if len(self.loss) else ''
            self.loss = []
            self.print_current_state(f'{p_frame}{p_loss}')
            
            # TODO for now it is only splitting the scene in batches and considering each batch as a separate scene
            # make it so that the gradients get computed for the entire scene and backpropagated by just computing everything until the end without gradients and then
            # going back computing and propagating gradients at each batch (last batch, second last, ...)
            
            self._run_pass(scene_batch)
        
        # TODO ???
        # Saving up memory for next scene
        del scene_batch
        gc.collect()
        #torch.cuda.empty_cache() # Not needed
