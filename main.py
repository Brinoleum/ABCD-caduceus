from transformers import (AutoTokenizer, 
                          Trainer, 
                          TrainingArguments,
                          get_cosine_schedule_with_warmup,
                          )
from model import CaduceusOrdinalRegressor
from metrics import compute_ordinal_metrics
from bitsandbytes.optim import AdamW
from dataloader import SNPDataset
from torch.utils.data import random_split
import os
os.environ["WANDB_PROJECT"] = "ABCD-caduceus"


def main():
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = CaduceusOrdinalRegressor(model_name)
    data = SNPDataset(tokenizer)

    '''
    following training regimen in the paper:
    - change optimizer and scheduler to cosine annealing
    - batch size = 2^20 tokens per batch with a sequence length of 16569bp ~= 63
    '''
    optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=0.01, betas=(0.95, 0.9), )
    scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,           # about 5-10% of the total training steps
            num_training_steps=1166
            )
    args = TrainingArguments(
            output_dir="./caduceus-output",
            run_name="nikola-single-gpu+AdamW+Cosine",
            per_device_train_batch_size=4, # make it a nice round power of 2
            gradient_accumulation_steps=4,
            num_train_epochs=5,
            save_steps=500,
            save_safetensors=False,         # model is setup such that safetensors can't save properly
            bf16=True,
            max_grad_norm=1.0,              # gradient clipping to prevent NaN
            report_to="wandb",
            eval_strategy="epoch",
            logging_steps=1,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_mode="default",
            )

    train, test = random_split(data, [0.8, 0.2])

    trainer = Trainer(model,
                      args=args,
                      train_dataset = train, 
                      eval_dataset = test,
                      optimizers=(optimizer, scheduler),
                      compute_metrics=compute_ordinal_metrics
                      )
    trainer.train()
    

if __name__ == "__main__":
    main()
