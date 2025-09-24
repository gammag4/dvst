import torch
import torch.amp as amp


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
            'skipped': self._skipped,
            **super().state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.batch_retry_enabled = state_dict.pop('batch_retry_enabled')
        self.max_retries = state_dict.pop('max_retries')
        self.should_retry_batch = state_dict.pop('should_retry_batch')
        self.num_retries = state_dict.pop('num_retries')
        self._skipped = state_dict.pop('skipped')
        return super().load_state_dict(state_dict)
