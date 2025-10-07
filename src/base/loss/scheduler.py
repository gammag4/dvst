from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable


TLoss = TypeVar('TLoss', bound=Callable)


class LossScheduler(ABC, Generic[TLoss]):
    def __init__(self, loss: TLoss, n_iter: int):
        self.loss = loss
        self.n_iter = n_iter
        self.iter = 0
        self._update_loss()
    
    @abstractmethod
    def _update_loss(self):
        pass
    
    def step(self):
        self.iter += 1
        self._update_loss()
    
    def state_dict(self):
        return {
            'n_iter': self.n_iter,
            'iter': self.iter
        }
    
    def load_state_dict(self, state_dict):
        self.n_iter = state_dict['n_iter']
        self.iter = state_dict['iter']


class LambdaLossScheduler(LossScheduler[TLoss]):
    def __init__(self, loss: TLoss, n_iter: int, loss_lambda: Callable[[TLoss, int, int], None]):
        super().__init__(loss, n_iter)
        self.loss_lambda = loss_lambda
        self.loss_lambda(self.loss, self.n_iter, self.iter)
    
    def _update_loss(self):
        self.loss_lambda(self.loss, self.n_iter, self.iter)
