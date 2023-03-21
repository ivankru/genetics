import itertools
import torch
from utils.datasets import onehot_to_str


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


def generator_stats(generator, z_size, condition, k_mers=4, n_trials=1000):
    kmers_false = kmers_dict(k_mers)
    device = condition.device
    for i in range(n_trials):
        z = torch.randn(1, z_size).to(device)
        condition = torch.randint(2, (1,)).to(device)
        fake = generator(z, condition).squeeze(0).transpose(1,0)
        dna_fake = onehot_to_str(fake.cpu().detach().numpy())
        false_dict = kmer_stat(dna_fake, kmers_false)

    number_of_kmers = sum(false_dict.values())
    for key in false_dict:
        false_dict[key] = false_dict[key] / number_of_kmers

    return false_dict


def dataset_stats(train_loader, k_mers=4, n_trials=None):
    kmers = kmers_dict(k_mers)

    for i, batch in enumerate(train_loader):
        if n_trials is not None:
            if i > n_trials:
                break
        _, labels, sequences, _, _ = batch
        sequences = sequences.squeeze(0).numpy()
        dna = onehot_to_str(sequences)
        kmers_dict = kmer_stat(dna, kmers)
        
    number_of_kmers = sum(kmers_dict.values())
    for key in kmers_dict:
        kmers_dict[key] = kmers_dict[key] / number_of_kmers
    return kmers_dict

