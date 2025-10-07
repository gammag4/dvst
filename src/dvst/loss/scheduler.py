import torch

from src.base.loss import LossScheduler

from .loss import PerceptualLoss


# beta tells which minimum value a weight can have (the maximum is 1), its range is [0, 1]
# n_iter is the number of iterations
# i is the current iteration, its range is [0, n_iter - 1]
# perc_idx are the indices of each layer from the cnn where the output is extracted from (the original image also counts as the output of the layer 0)
def h(n_iter, i, perc_idx: torch.Tensor):
    return (i / (n_iter - 1) - 0.5) * torch.cos((torch.pi * perc_idx) / (perc_idx.shape[0] - 1)) + 0.5


def r1(beta, n_iter, i, perc_idx):
    return (1 - beta) * h(n_iter, i, perc_idx) + beta


def r2(beta, n_iter, i, perc_idx):
    return torch.ones(perc_idx.shape, device=perc_idx.device)


def r3(beta, n_iter, i, perc_idx):
    return (beta - 1) * h(n_iter, i, perc_idx) + 1


class PerceptualLossScheduler(LossScheduler[PerceptualLoss]):
    # Regime can be either:
    #   'constant': Weights are always constant
    #   'deep_to_shallow': Starts giving more weights to deeper layers and gradually goes to givin more weights to shallower ones in the end
    #   'shallow_to_deep': Starts giving more weights to shallower layers and gradually goes to givin more weights to deeper ones in the end
    def __init__(self, loss: PerceptualLoss, n_iter: int, beta: float=0.5, regime: str='constant'):
        self.beta = beta

        if regime == 'deep_to_shallow':
            r = r1
        elif regime == 'constant':
            r = r2
        elif regime == 'shallow_to_deep':
            r = r3
        else:
            raise Exception(f'Regime "{regime}" not supported for perceptual loss scheduler')
        
        self.r = r
        super().__init__(loss, n_iter)
    
    def _update_loss(self):
        perc_idx = torch.arange(self.loss.layer_weights.shape[0], device=self.loss.layer_weights.device)
        self.loss.layer_weights = self.r(self.beta, self.n_iter, self.iter, perc_idx)
