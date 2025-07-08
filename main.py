from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataloader import SNPDataset


def main():
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
    data = SNPDataset(tokenizer)

    

if __name__ == "__main__":
    main()
