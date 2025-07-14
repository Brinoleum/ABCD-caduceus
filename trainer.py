from transformers import Trainer, get_cosine_schedule_with_warmup
from bitsandbytes.optim import AdamW

class CaduceusTrainer(Trainer):
    """
    - for FSDP, overrides the create_scheduler and create_optimizer methods
    - implements AdamW and cosine decay
    """

    def create_scheduler(self, num_training_steps, optimizer=None):
        return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=500,
                num_training_steps=int(0.8*11648)
               )

    def create_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=8e-3, weight_decay=0.01, betas=(0.95, 0.9))
        return self.optimizer
