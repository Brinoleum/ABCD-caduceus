from transformers import Trainer, get_cosine_schedule_with_warmup
from bitsandbytes.optim import AdamW

class CaduceusTrainer(Trainer):
    """
    - for FSDP, overrides the create_scheduler and create_optimizer methods
    """

    '''
    following training regimen in the paper:
    - change optimizer and scheduler to cosine annealing
    - batch size = 2^20 tokens per batch with a sequence length of 16569bp ~= 63
    '''
    def create_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=8e-3, weight_decay=0.01, betas=(0.95, 0.9))
        return self.optimizer
    
    def create_scheduler(self, num_training_steps, optimizer=None):
        self.lr_scheduler = get_cosine_schedule_with_warmup(
                                optimizer,
                                num_warmup_steps=0.1 * num_training_steps,
                                num_training_steps=num_training_steps
                            )
        return self.lr_scheduler

    # no custom loss function for ordinal regression, 
    # should already be included in the model implementation

    """
    weighted sampling for imbalanced phenotypes
    # of each class:
    {0: 5708, 1: 3467, 2: 1387, 4: 734, 3: 555}

    due to the nature of ordinal regression it may or may not be necessary to address imbalanced classes since it should model cumulative probabilities
    """

