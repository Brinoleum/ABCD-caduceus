from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          )

# ignoring ordinal regression for now, focusing on model correctness
# from model import CaduceusOrdinalRegressor
# from metrics import compute_ordinal_metrics
from trainer import CaduceusTrainer
from dataloader import SNPDataset, FakeDataset
from torch.utils.data import random_split
from torch.nn.init import xavier_normal_
from peft import LoraConfig, get_peft_model

import os
os.environ["WANDB_PROJECT"] = "ABCD-caduceus"


def main():
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data = SNPDataset(tokenizer)
    
    # sanity check to see if the data converges
    # data = FakeDataset(tokenizer)

    args = TrainingArguments(
            # general config options
            output_dir="./caduceus-output",
            run_name="dummy-test",
            
            # general training options
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=5,
            save_steps=500,
            save_safetensors=False,         # model is setup such that safetensors can't save properly
            max_grad_norm=1.0,              # gradient clipping to prevent NaN
            
            # optimizer/scheduler
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            learning_rate=8e-3,
            
            # logging
            # report_to="wandb",
            eval_strategy="epoch",
            logging_steps=1,

            # hardware specific
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_mode="default",
            )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.01,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["x_proj", "in_proj", "out_proj", "dt_proj"]
    )

    train, test = random_split(data, [0.8, 0.2])
    
    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, trust_remote_code=True)
        xavier_normal_(model.score.weight)
        return get_peft_model(model, lora_config)
    
    trainer = CaduceusTrainer(model=None,
                              args=args,
                              train_dataset=train, 
                              eval_dataset=test,
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
    
    # best_trials = trainer.hyperparameter_search(
    #     direction="maximize",
    #     backend="optuna",
    #     hp_space=hp_space,
    #     n_trials=20,
    #     compute_objective=compute_objective
    # )

    # print(f"Optimized Parameters: {best_trials.hyperparameters}")
    trainer.train()

if __name__ == "__main__":
    main()
