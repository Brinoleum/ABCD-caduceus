import numpy as np
import pandas as pd
import torch
from pandas_plink import read_plink1_bin
from pyliftover import LiftOver
from pyfaidx import Fasta

snp_file    = "data/plink/ABCD_202209.updated.nodups.curated.cleaned_indivs.bed"
phenotypes  = "data/pheno/mh_y_ksads_gad.csv" # placeholder for now, using the KSADS anxiety metric
genome_file = "data/hg38/hg38.fa"
mtChr       = "NC_012920.1" # label from NCBI hg38 dataset


G = read_plink1_bin(snp_file)
P = pd.read_csv(phenotypes)
L = LiftOver("hg19", "hg38")
F = Fasta(genome_file)
mtDna = F[mtChr][:]

def get_samples():
    return G.sample.to_numpy()

@np.vectorize
def convert_coordinate(coord):
    converted = L.convert_coordinate("chrM", coord)
    
    try:
        assert converted is not None and len(converted) == 1
    except:
        return -1

    _, out, _, _ = converted[0]
    return out

def read_variants(sample):
    table = G.where(G.chrom=="26", drop=True).where(G.sample==sample, drop=True)
    # a1 is the reference in the plink file, indices offset by 1
    # coordinate N in the table converts to N-1 in liftover
    # skip invalid coordinate conversions
    data = np.array([convert_coordinate(table.pos.to_numpy()-1), table.a0.to_numpy()]).T
    out = np.array(list(mtDna.seq))
    data = data[data[:, 0] != -1]
    out[data[:, 0].astype(int)] = data[:, 1]
    return "".join(out)

from torch.utils.data import Dataset

class SNPDataset(Dataset):
    def __init__(self, tokenizer):
        samples = get_samples()
        self.phenos = P.groupby("src_subject_id")["ksads_gad_raw_273_t"].max().fillna(0).astype(int).to_dict()
        self.tokenizer = tokenizer
        
        # align pheno ids with plink ids - ids referenced in the plink are sometimes not in the csv
        self.samples = samples[np.isin(samples, np.array(list(self.phenos.keys())))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(read_variants(sample), return_tensors='pt')
        return {"input_ids": encoding["input_ids"].flatten(), "labels": torch.tensor(self.phenos[sample], dtype=torch.long)}

