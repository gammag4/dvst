import gc
from typing import cast

from src.base.run import DefaultDistributedTrainer

from src.dvst.config import *
from src.dvst.model import DVST


class DVSTTrainer(DefaultDistributedTrainer[DVSTDatasetConfig, DVSTModelConfig, DVSTOptimizerConfig, DVST]):
    def __init__(self, config, dataset_provider, model_provider, optimizer_provider, log_provider):
        super().__init__(config, dataset_provider, model_provider, optimizer_provider, log_provider)
        
        self.current_scene_frame = None
        self.current_scene_n_frames = None
    
    def _run_forward(self, *args):
        scene_batch, = args
        loss, _ = self.model(scene_batch)
        return loss
    
    def _run_dataset_batch(self, batch):
        model = cast(DVST, self.model.module)
        
        scene = batch.load_scene(model.scene_batch_size, self.device)
        
        self.logger.log({'scene_id': scene.scene_id})
        
        for i, scene_batch in enumerate(scene):
            # TODO save each batch history at checkpoints too put to separate function
            #   in same function also save latent_embeds data sum mean var add it explicitly in function as extra args
            #   add everything and log it to a file
            #   then visualize everything with a notebook
            
            self.current_scene_frame = i * scene.batch_size
            self.current_scene_n_frames = scene.n_frames
            
            self.logger.log({'frame': self.current_scene_frame})
            
            # TODO for now it is only splitting the scene in batches and considering each batch as a separate scene
            # make it so that the gradients get computed for the entire scene and backpropagated by just computing everything until the end without gradients and then
            # going back computing and propagating gradients at each batch (last batch, second last, ...)
            
            self._run_pass(scene_batch)
        
        self.current_scene_frame = None
        self.current_scene_n_frames = None
        
        # TODO ???
        # Saving up memory for next scene
        del scene_batch
        gc.collect()
        #torch.cuda.empty_cache() # Not needed
