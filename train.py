import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from joblib import load
from utils.datasets import Dataset, DNATokenizer
from utils.train_val import train_iter, validation, \
    input_part, descriminator_part, generator_part, \
    end_of_gan_epoch, generator_part_wasser, \
    descriminator_part_wasser
from utils.augmentation import augmentation_func
from utils.model import DNAdescriminator, \
     DNAgenerator, DNAgenerator2
from transformers import BertForSequenceClassification, BertConfig


def import_data(MODEL_NUMBER, batch_size, augmentation=None):
    folder_path = "/gim/lv01/dumerenkov/zdna_data/datasets"
    #file_name = f'ds_w_seq_hg_fold{MODEL_NUMBER}.pkl'
    #file_name = f'ds_w_seq_hg_fold{MODEL_NUMBER}_kouzine.pkl'
    #file_name = f'ds_w_seq_mm_fold{MODEL_NUMBER}_chipseq.pkl'
    file_name = f'ds_w_seq_mm_fold{MODEL_NUMBER}_kouzine.pkl'
    file_path = os.path.join(folder_path, file_name)

    tokenizer = DNATokenizer.from_pretrained('/home/kruzhilov/genetics/6-new-12w-0/')

    file_path = os.path.join(folder_path, file_name)
    print("file path:", file_path)
    train_dataset, test_dataset = load(file_path)

    #augmentation = augmentation_func()

    train_dataset = Dataset(train_dataset.chroms, \
         train_dataset.features, train_dataset.dna_source,\
         train_dataset.features_source, train_dataset.labels_source,\
         train_dataset.intervals, train_dataset.tokenizer,\
         augmentation=None)
    weights = torch.FloatTensor(train_dataset.samples_weight)
    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
         sampler=sampler, num_workers=0)

    test_dataset = Dataset(train_dataset.chroms,\
         test_dataset.features, test_dataset.dna_source,\
             train_dataset.features_source, test_dataset.labels_source,\
             test_dataset.intervals, test_dataset.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    return train_loader, test_loader
 

def train_z_classifier(epoch_number, batch_size, train_loader, test_loader):
    # config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
    # dir_to_pretrained_model = "6-new-12w-0/"
    # net = BertForSequenceClassification.from_pretrained(dir_to_pretrained_model, config=config)
    net = DNAclassificator()
    net.to(device)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0005, weight_decay=0.05)

    for epoch in range(epoch_number):
        loss_list = []
        for idx, batch in enumerate(train_loader):
            train_iter(batch, net, loss_func, optimizer, loss_list, \
                       device, gradient_clipping=True, \
                       berta=False, kmer=False)
            
            if idx % (1000 // batch_size) == 0:
                mean_loss = sum(loss_list) / len(loss_list)
                loss_list = []
                rocauc, rep_dict, mean_val_loss = \
                    validation(test_loader, net, loss_func, \
                               device, berta=False, kmer=False)
                formated_string = ("{epoch} train loss:{tr_loss:.4f}, val_loss:{val_loss:.4f}, " + \
                    "f1_0:{f1_0:.2f}%, f1_1:{f1_1:.2f}%, " + \
                    "rocauc:{rocauc:.2f}%").format(epoch=epoch, tr_loss=mean_loss, \
                f1_0=100*rep_dict["0"]["f1-score"], f1_1=100*rep_dict["1"]["f1-score"], rocauc=100*rocauc, val_loss=mean_val_loss)
                print(formated_string) #"val_loss:", mean_loss_val,


    real_loss = d_result_real.sum()
    real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
    r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
    loss = r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def train_gan(epoch_number, train_loader, device, gradient_clipping=False):
    z_size = 64
    path_to_save = "/ayb/vol1/kruzhilov/weights/genomes/"
    generator = DNAgenerator2(z_size)
    #generator.load_state_dict(torch.load(os.path.join(path_to_save, "generator.pth")))
    generator.to(device)
    descriminator = DNAdescriminator()
    #descriminator.load_state_dict(torch.load(os.path.join(path_to_save, "descriminator.pth")))
    descriminator.to(device)

    gen_optim = torch.optim.Adam(generator.parameters(), lr=0.00006, betas=(0.5, 0.99))
    desc_optim = torch.optim.AdamW(descriminator.parameters(), lr=0.00006, weight_decay=0.0)
    gen_loss_list = []
    desc_loss_list = []
    grad_penalty_list = []
    frequency_loss_list = []
    batch_diversity_list = []

    for epoch in range(epoch_number):
        for idx, batch in enumerate(train_loader):
            input, labels = input_part(batch, kmer=False, device=device)
            labels = labels.to(device)

            for i in range(5):
                descr_loss = descriminator_part(descriminator, generator, input,\
                    z_size, desc_loss_list, grad_penalty_list, labels, \
                    gradient_penalty_coeff=0.2)    
                desc_optim.zero_grad()
                descr_loss.backward()
                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(parameters=descriminator.parameters(), max_norm=0.2)
                desc_optim.step()

            gen_loss = generator_part(descriminator, generator, input, z_size,\
                 gen_loss_list, batch_diversity_list, frequency_loss_list, labels)    
            gen_optim.zero_grad()
            gen_loss.backward()
            # for name, param in generator.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is not None:
            #             grad = param.grad.abs().mean().item()
            #             print(name, grad)
            gen_optim.step()

            if idx % 100 == 0:
                gen_loss = sum(gen_loss_list) / len(gen_loss_list)
                descr_loss = sum(desc_loss_list) / len(desc_loss_list)
                gradient_penalty = sum(grad_penalty_list) / len(grad_penalty_list)
                batch_diversity = sum(batch_diversity_list) / len(batch_diversity_list)
                frequency_loss = sum(frequency_loss_list) / len(frequency_loss_list)
                formated_string = "{epoch} D:{descr:1.3f}, D_grad:{grad:1.3f}, G_diver:{diver:1.4f}, G_freq:{freq:1.3f}, G:{gen:1.3f}" \
                    .format(epoch=epoch, descr=descr_loss, grad=gradient_penalty, \
                        diver=batch_diversity, freq=frequency_loss, gen=gen_loss)
                print(formated_string)
                gen_loss_list = []
                desc_loss_list = []
                grad_penalty_list = []
            
        end_of_gan_epoch(generator, descriminator, input,\
             z_size, labels, path_to_save=path_to_save)


if __name__ == "__main__":
    device = "cuda:2"
    epoch_number = 150
    batch_size = 64

    MODEL_NUMBER = 0
    train_loader, test_loader = import_data(MODEL_NUMBER, batch_size)

    train_gan(epoch_number, train_loader, device)


