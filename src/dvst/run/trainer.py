import gc
from typing import cast

from src.base.run import DistributedTrainer

from src.dvst.config import *
from src.dvst.model import DVST


class DVSTTrainer(DistributedTrainer[DVSTDatasetConfig, DVSTModelConfig, DVSTOptimizerConfig, DVST]):
    def _run_forward(self, *args):
        scene_batch, = args
        loss, _ = self.model(scene_batch)
        return loss
    
    def _run_dataset_batch(self, batch):
        model = cast(DVST, self.model.module)
        
        scene = batch.load_scene(model.scene_batch_size, self.device)
        
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
