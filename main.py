from transformers import (AutoTokenizer, 
                          TrainingArguments,
                          Trainer
                          )
from model import CaduceusOrdinalRegressor
# subclassed trainer with hardcoded parameters not quite necessary for hyperparameter sweep
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
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            num_train_epochs=5,
            save_steps=500,
            save_safetensors=False,         # model is setup such that safetensors can't save properly
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
    
    trainer = Trainer(model=None,
                      args=args,
                      train_dataset=train, 
                      eval_dataset=test,
                      compute_metrics=compute_ordinal_metrics,
                      model_init=model_init
        )
    
    def hp_space(trial):
        args = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            # not sure how big of a batch we can support on the A40 w/ or w/o DDP/FSDP
            # but we can also mess around with the gradient accumulation
            # "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4]),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [4, 8, 16, 32]),
        }
        return args
    
    def compute_objective(metrics):
        """
        metrics include:
        MAE
        MSE
        RMSE
        Exact/within 1-2 accuracy
        Kendall Tau
        Spearman Rho
        Individual class accuracy

        eyeballing the weights but the general idea is to maximize accuracy and penalize error
        """
        return (metrics["eval_exact_accuracy"] 
                + 0.75*metrics["eval_within_1_accuracy"] 
                + 0.5*metrics["eval_within_2_accuracy"]
                - metrics["eval_mae"]
                - metrics["eval_mse"]
        )
    
    best_trials = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=20,
        compute_objective=compute_objective
    )

    print(f"Optimized Parameters: {best_trials.hyperparameters}")

if __name__ == "__main__":
    main()