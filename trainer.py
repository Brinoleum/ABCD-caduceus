from transformers import Trainer, get_cosine_schedule_with_warmup
from bitsandbytes.optim import AdamW
import torch
import torch.nn.functional as F

class CaduceusTrainer(Trainer):
    """
    - for FSDP, overrides the create_scheduler and create_optimizer methods
        - setting this aside for now as we're doing a hyperparameter sweep to determine best config for the scheduler/optimizer
    - implements a custom loss function to account for class imbalances
        - weighted/focal loss focuses on the less represented classes
    """
    # def create_optimizer(self):
    #     self.optimizer = AdamW(self.model.parameters(), lr=8e-3, weight_decay=0.01, betas=(0.95, 0.9))
    #     return self.optimizer
    
    # def create_scheduler(self, num_training_steps, optimizer=None):
    #     self.lr_scheduler = get_cosine_schedule_with_warmup(
    #                             optimizer,
    #                             num_warmup_steps=0.1 * num_training_steps,
    #                             num_training_steps=num_training_steps
    #                         )
    #     return self.lr_scheduler

    """
    weighted sampling for imbalanced phenotypes
    # of each class:
    {0: 5708, 1: 3467, 2: 1387, 4: 734, 3: 555}
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        implementation of focal loss with default parameters alpha = 0.25, gamma = 2
        """
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        alpha, gamma = 0.25, 2 # TODO: these are defaults, we should add these to hyperparameter sweep
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt) ** gamma * ce_loss
        loss = focal_loss.mean()

        return (loss, outputs) if return_outputs else loss