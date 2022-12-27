import os
import numpy as np
from utils.datasets import Dataset, DNATokenizer, onehot_to_str
import itertools
from torch.utils.data import DataLoader
from joblib import load
import json
from utils.model import DNAgenerator
import torch
from utils.dna_stats import generator_stats

#A=0.235 C=0.172 T=0.415 G=0.178



def kmers_dict(k):
   mol_list = ["A", "C", "T", "G"]
   kmers = list(map(list, itertools.product(mol_list, repeat=k)))
   kmers = list(map(lambda x: "".join(x), kmers))
   zero_list = [0] * len(kmers)
   kmers_dict = dict(zip(kmers, zero_list))
   return kmers_dict


def kmer_stat(dna, kmers_dict):
    k = len(list(kmers_dict.keys())[0])
    for i in range(len(dna) - k):
        substr = dna[i:i+k]
        kmers_dict[substr] = kmers_dict[substr] + 1
    return kmers_dict


if __name__ == "__main__":
    MODEL_NUMBER = 1
    folder_path = "/gim/lv01/dumerenkov/zdna_data/datasets"
    file_name = f'ds_w_seq_hg_fold{MODEL_NUMBER}.pkl'
    file_path = os.path.join(folder_path, file_name)

    tokenizer = DNATokenizer.from_pretrained('/home/kruzhilov/genetics/6-new-12w-0/')

    file_path = os.path.join(folder_path, file_name)
    print("file path:", file_path)
    train_dataset, test_dataset = load(file_path)

    train_dataset = Dataset(train_dataset.chroms, \
         train_dataset.features, train_dataset.dna_source,\
         train_dataset.features_source, train_dataset.labels_source,\
         train_dataset.intervals, train_dataset.tokenizer,\
         augmentation=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    with open("weights/hh_stat.json", "r") as outfile:
        hh_kmer_dict = json.load(outfile)

    device = "cuda:1"
    z_size = 64
    path_to_save = "/home/kruzhilov/genetics/weights/"
    generator = DNAgenerator(z_size)
    generator.load_state_dict(torch.load(os.path.join(path_to_save, "generator.pth")))
    generator.to(device)
   
    cond = torch.FloatTensor(1).to(device)
    gen_k_mers = generator_stats(generator, z_size, cond, 1)
    aaa = 666
