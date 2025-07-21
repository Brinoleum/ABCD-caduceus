from transformers import AutoModel
import torch.nn as nn
import torch

"""
instead of using the AutoModelForSequenceClassification,
replace the classification head with an ordinal regression head, specifically proportional odds
"""

class CaduceusOrdinalRegressor(nn.Module):
    def __init__(self, model_name, num_classes=5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.num_classes = num_classes

        # single linear layer with num_classes-1 thresholds
        self.classifier = nn.Linear(512, num_classes - 1)
    
    def forward(self, input_ids, labels=None):
        outputs = self.backbone(input_ids=input_ids)
        pooled_output = outputs.last_hidden_state[:, 0] # Use [CLS] token
        
        # get threshold logits
        logits = self.classifier(pooled_output)
        
        # convert to class probs
        probabilities = torch.sigmoid(logits)

        if labels is not None:
            loss = self.ordinal_loss(probabilities, labels)
            return {"loss": loss, "logits": logits, "probabilities": probabilities}
        
        return {"logits": logits, "probabilities": probabilities}
    
    def ordinal_loss(self, probabilities, labels):
        # Create binary targets for each threshold
        batch_size = labels.size(0)
        targets = torch.zeros(batch_size, self.num_classes - 1).to(labels.device)
        
        for i in range(batch_size):
            targets[i, :labels[i]] = 1
        
        # Binary cross-entropy loss for each threshold
        loss = nn.functional.binary_cross_entropy_with_logits(probabilities, targets)
        return loss