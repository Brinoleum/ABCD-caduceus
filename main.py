from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from dataloader import SNPDataset
from torch.utils.data import random_split
import wandb
import os
os.environ["WANDB_PROJECT"] = "ABCD-caduceus"


def main():
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
    data = SNPDataset(tokenizer)
    args = TrainingArguments(
            output_dir="./caduceus-output",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            save_steps=500,
            report_to="wandb",
            logging_steps=1,
            torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_mode="default",
            )

    train, test = random_split(data, [0.8, 0.2])

    trainer = Trainer(model,
                      args=args,
                      train_dataset = train, 
                      eval_dataset = test)
    trainer.train()
    

if __name__ == "__main__":
    main()
