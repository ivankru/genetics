import os
import numpy as np
from utils.datasets import Dataset, DNATokenizer, onehot_to_str
import itertools
from torch.utils.data import DataLoader
from joblib import load
import json
from utils.model import DNAgenerator2
import torch
from utils.dna_stats import generator_stats, kmers_dict, kmer_stat
from transformers import BertForSequenceClassification, BertConfig, PreTrainedTokenizer, BasicTokenizer, BertForTokenClassification
from dima_test import classification_metrics, seq2kmer

#hh chipseq A=0.235 C=0.172 T=0.415 G=0.178
#kusin mm kouzine 'A':0.2733 'C':0.2296 'T':0.2284 'G':0.2685


def random_dna(n=512):
    mol_list = np.array(["A", "C", "T", "G"])
    dna = np.random.choice(mol_list, n)
    dna = "".join(dna)
    return dna


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, generator, tokenizer, z_size, dataset_size=1000):
        self.dataset_size = dataset_size
        self.tokenizer = tokenizer
        self.generator = generator
        self.generator.eval()
        self.z_size = 64

    def __getitem__(self, idx):
        z_label = torch.LongTensor([1])
        z = torch.randn(1, self.z_size) 
        with torch.no_grad():
            dna = self.generator(z, z_label).squeeze(0).cpu()
        dna = onehot_to_str(dna.numpy().transpose())
        k_mers = seq2kmer(dna, 6)
        encoded_k_mers = self.tokenizer.encode_plus(k_mers, add_special_tokens=False, max_length=512)["input_ids"]
        encoded_k_mers = torch.LongTensor(encoded_k_mers)
        return z_label, z_label, encoded_k_mers, encoded_k_mers, encoded_k_mers

    def __len__(self):
        return self.dataset_size


def test_z_dna_gan(z_size=64):
    device = "cpu:1"
    generator = DNAgenerator2(z_size)
    generator.load_state_dict(torch.load("/ayb/vol1/kruzhilov/weights/genomes/generator.pth"))
    tokenizer = DNATokenizer.from_pretrained('/home/kruzhilov/genetics/6-new-12w-0/')
    config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
    dir_to_pretrained_model = "6-new-12w-0/"
    classificator = BertForSequenceClassification.from_pretrained(dir_to_pretrained_model, config=config)
    classificator.load_state_dict(torch.load("/ayb/vol1/kruzhilov/weights/genomes/dna_bert.pth"))

    gan_dataset = GeneratedDataset(generator, tokenizer, z_size)
    test_loader = DataLoader(gan_dataset, batch_size=1, num_workers=0)

    rocauc, rep_dict, _ = classification_metrics(test_loader, classificator, torch.nn.BCEWithLogitsLoss(), device)


if __name__ == "__main__":
    test_z_dna_gan()
    # MODEL_NUMBER = 1
    # folder_path = "/gim/lv01/dumerenkov/zdna_data/datasets"
    # file_name = f'ds_w_seq_mm_fold{MODEL_NUMBER}_kouzine.pkl'
    # file_path = os.path.join(folder_path, file_name)

    # tokenizer = DNATokenizer.from_pretrained('/home/kruzhilov/genetics/6-new-12w-0/')

    # file_path = os.path.join(folder_path, file_name)
    # print("file path:", file_path)
    # train_dataset, test_dataset = load(file_path)

    # train_dataset = Dataset(train_dataset.chroms, \
    #      train_dataset.features, train_dataset.dna_source,\
    #      train_dataset.features_source, train_dataset.labels_source,\
    #      train_dataset.intervals, train_dataset.tokenizer,\
    #      augmentation=None)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    # k_mers = 4
    # n_trials = None
    # kmer = kmers_dict(k_mers)
    # for idx, batch in enumerate(train_loader):
    #     _, labels, sequences, _, _ = batch
    #     sequences = sequences.squeeze(0).numpy()
    #     dna = onehot_to_str(sequences)
    #     kmer_stat(dna, kmer)
    #     if n_trials:
    #         if idx > n_trials:
    #             break

    # number_of_kmers = sum(kmer.values())
    # for key in kmer:
    #     kmer[key] = kmer[key] / number_of_kmers

    # with open("weights/kouzine_mm_stat.json", "w") as outfile:
    #     json.dump(kmer, outfile)

    # with open("weights/kouzine_mm_stat.json", "r") as outfile:
    #     hh_kmer_dict = json.load(outfile)


    #generator frequences
    # k_mers = 4
    # n_trials = 2000
    # kmers_false = kmers_dict(k_mers)
    # device = "cuda:1"
    # z_size = 64
    # path_to_save = "/home/kruzhilov/genetics/weights/"
    # generator = DNAgenerator(z_size)
    # generator.load_state_dict(torch.load(os.path.join(path_to_save, "generator.pth")))
    # generator.to(device)
   
    # cond = torch.FloatTensor(1).to(device)
    # gen_k_mers = generator_stats(generator, z_size, cond, 1)
    
    #random dna
    # k_mers = 4
    # n_trials = 4000
    # kmers_false = kmers_dict(k_mers)
    # for i in range(n_trials):
    #     dna_fake = random_dna()
    #     false_dict = kmer_stat(dna_fake, kmers_false)

    # number_of_kmers = sum(false_dict.values())
    # for key in false_dict:
    #     false_dict[key] = false_dict[key] / number_of_kmers

    # delta_p = [false_dict[key] - hh_kmer_dict[key] for key in hh_kmer_dict]
    # delta_p = np.abs(delta_p).sum() / 2
    # print("4mers discrepency:", delta_p)
