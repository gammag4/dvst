import gc
from typing import cast

from src.base.run import DefaultDistributedTrainer

from src.dvst.datasets.scene_dataset import SceneDataset
from src.dvst.config import *
from src.dvst.loss import PerceptualLoss
from src.dvst.model import DVST


class DVSTTrainer(DefaultDistributedTrainer[DVSTDatasetConfig, DVSTModelConfig, DVSTOptimizerConfig, DVSTLossConfig, DVST]):
    @property
    def n_train_steps(self):
        dataset = cast(SceneDataset, self.train_data.dataset)
        
        return dataset.n_frames // self.config.model.scene_batch_size # TODO
    
    @property
    def n_val_steps(self):
        dataset = cast(SceneDataset, self.val_data.dataset)
        
        return dataset.n_frames // self.config.model.scene_batch_size # TODO
    
    def load_default_state(self):
        super().load_default_state()
        
        self.current_scene_frame = 0
        self.last_frames = None
    
    def state_dict(self):
        state_dict = super().state_dict()
        
        state_dict['current_scene_frame'] = self.current_scene_frame
        state_dict['last_frames'] = self.last_frames
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        
        self.current_scene_frame = state_dict['current_scene_frame']
        self.last_frames = state_dict['last_frames']
    
    def _run_forward(self, *args):
        scene_batch, = args
        latent_embeds = None # TODO
        loss, latent_embeds, self.last_frames = self.model(scene_batch, latent_embeds)
        
        return loss
    
    def _step(self):
        if isinstance(self.base_model.loss, PerceptualLoss):
            loss = cast(PerceptualLoss, self.base_model.loss)
            self.logger.log({'perceptual_weights': loss.layer_weights})
        
        super()._step()
    
        self.current_scene_frame += self.current_scene_batch_size
    
    def _run_dataset_batch(self, batch):
        scene = batch.load_scene(self.base_model.scene_batch_size, self.device)
        
        self.current_scene_batch_size = scene.batch_size
        
        self.logger.log({'scene_id': scene.scene_id})
        
        for scene_batch in scene:
            # TODO save each batch history at checkpoints too put to separate function
            #   in same function also save latent_embeds data sum mean var add it explicitly in function as extra args
            #   add everything and log it to a file
            #   then visualize everything with a notebook
            
            self.logger.log({'frame': self.current_scene_frame})
            
            # TODO for now it is only splitting the scene in batches and considering each batch as a separate scene
            # make it so that the gradients get computed for the entire scene and backpropagated by just computing everything until the end without gradients and then
            # going back computing and propagating gradients at each batch (last batch, second last, ...)
            
            self._run_pass(scene_batch)
        
        self.current_scene_frame = 0
        
        # TODO ???
        # Saving up memory for next scene
        del scene_batch
        gc.collect()
        #torch.cuda.empty_cache() # Not needed
