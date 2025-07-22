from transformers import (AutoTokenizer, 
                          TrainingArguments,
                          )
from model import CaduceusOrdinalRegressor
from trainer import CaduceusTrainer
from metrics import compute_ordinal_metrics
from dataloader import SNPDataset
from torch.utils.data import random_split
import os
os.environ["WANDB_PROJECT"] = "ABCD-caduceus"


def main():
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data = SNPDataset(tokenizer)

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
    
    def model_init():
        return CaduceusOrdinalRegressor(model_name)
    
    trainer = CaduceusTrainer(model=None,
                              args=args,
                              train_dataset = train, 
                              eval_dataset = test,
                              compute_metrics=compute_ordinal_metrics,
                              model_init=model_init
                      )
    
    def wandb_hp_space(trial):
        return {
                "method": "bayes",
                "metric": {"name": "eval_loss", "goal": "minimize"},
                "parameters": {
                        "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
                        "per_device_train_batch_size": {"values": [1, 2, 4, 8]}
                }}
    
    best_trials = trainer.hyperparameter_search(
        direction=["minimize", "maximize"],
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=20,
    )

    print(f"Optimized Parameters: {best_trials.hyperparameters}")

if __name__ == "__main__":
    main()
