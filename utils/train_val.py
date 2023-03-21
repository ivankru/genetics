import numpy as np
import os
from sklearn.metrics import roc_auc_score, classification_report
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import json
import sys
 
sys.path.insert(0, '/home/kruzhilov/genetics')

from utils.datasets import onehot_to_str
from utils.dna_stats import generator_stats


def input_part(batch, kmer, device):
    if kmer:
        _, labels, _, input_ids, _ = batch
        input_ids = input_ids.type(torch.LongTensor)
        input = input_ids.to(device)
    else:
        _, labels, sequences, _, _ = batch
        sequences = torch.transpose(sequences, 1, 2) 
        sequences = sequences.type(torch.FloatTensor)
        input = sequences.to(device)

    labels = labels.type(torch.FloatTensor)
    labels = labels.to(device)    
    labels = labels.sum(dim=1)
    labels[labels > 0] = 1
    return input, labels


def prediction_part(model, input, loss_func, labels, berta):
    prediction = model(input)
    if berta:
        prediction = prediction["logits"]
    prediction = prediction.diff(dim=1) #["logits"]
    prediction = prediction.squeeze(1)
    loss = loss_func(prediction, labels)
    return loss, prediction


def train_iter(batch, model, loss_func, optimizer, \
               loss_list, device, \
               berta=False, gradient_clipping=False, kmer=True):
    optimizer.zero_grad()
    input, labels = input_part(batch, kmer, device)
    loss, _ = prediction_part(model, input, loss_func, labels, berta)

    loss_list.append(loss.item())
    # gradient clipping
    if gradient_clipping:
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.1)
    loss.backward()
    optimizer.step()
   

def validation(data_loader, net, loss_func, \
                          device, berta=False, kmer=True):
    net.eval()
    label_list = None
    prediction_list = None
    loss_list = []
    with torch.no_grad():
        for batch in data_loader:
            input, labels = input_part(batch, kmer, device)            
            loss, prediction = prediction_part(net, input, loss_func, labels, berta)

            labels = labels.type(torch.LongTensor)
            loss_list.append(loss.item())
            prediction = prediction.detach().cpu().numpy()
            if label_list is not None:
                label_list = np.hstack([label_list, labels.detach().cpu().numpy()])
            else:
                label_list = labels.detach().cpu().numpy()
            if prediction_list is not None:
                prediction_list = np.hstack([prediction_list, prediction])
            else:
                prediction_list = prediction
    rocauc = roc_auc_score(label_list, prediction_list)
    prediction_list = torch.sigmoid(torch.FloatTensor(prediction_list))
    prediction_list = np.round(prediction_list.numpy())
    rep_dict = classification_report(label_list, prediction_list, output_dict=True)
    mean_loss = sum(loss_list) / len(loss_list)
    return rocauc, rep_dict, mean_loss


def discriminator_logistic_non_saturating(d_result_fake, d_result_real):
    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real)) / 2
    return loss.mean()


def generator_logistic_non_saturating(d_result_fake):
    return F.softplus(-d_result_fake).mean()


def zero_centered_gradient_loss(d_r, d_f, real_data, fake_data):
    grad_r = autograd.grad(d_r.sum(), real_data,
                      # grad_outputs=torch.ones_like(d_r),
                       allow_unused=True, 
                       create_graph=True, 
                       retain_graph=True)[0].view(d_r.size(0),-1)
    # grad_f = autograd.grad(d_f.sum(), fake_data,
    #                    #grad_outputs=torch.ones_like(d_f),
    #                    allow_unused=True, 
    #                    create_graph=True, 
    #                    retain_graph=True)[0].view(d_f.size(0),-1)
    loss_gp = torch.mean(grad_r.norm(dim=1,p=2)**2) #+ torch.mean(grad_f.norm(dim=1,p=2)**2)
    return loss_gp


def wasserstein_gradient_loss(descriminator, real_data, fake_data):
    alpha = torch.rand(1).to(real_data.device)
    mixed_data = alpha * real_data + (1 - alpha) * fake_data
    mixed_data.requires_grad_(True)
    d_mixed = descriminator(mixed_data)
    grad_m = autograd.grad(d_mixed.sum(), mixed_data,
                      # grad_outputs=torch.ones_like(d_r),
                       allow_unused=True, 
                       create_graph=True, 
                       retain_graph=True)[0].view(d_mixed.size(0),-1)
    
    grad_norm = grad_m.norm(dim=1,p=2)
    ones = torch.ones_like(grad_norm)
    loss_gp = torch.square(grad_norm - ones)
    loss_gp = loss_gp.mean()
    return loss_gp


def descriminator_part_wasser(descriminator, generator, real, z_size, \
                 desc_loss_list, penalty_list, label,\
                 gradient_penalty_coeff=0.1):
    device = label.device
    #descriminator true part
    #real.requires_grad_(True)
    descr_real = descriminator(real)
    #descriminator fake part
    with torch.no_grad():
        z = torch.randn(real.shape[0], z_size)
        z = z.to(device)
        fake = generator(z, label)
    #fake.requires_grad_(True)
    descr_fake = descriminator(fake)
    loss = -descr_real.mean() + descr_fake.mean()
    gradient_penalty = wasserstein_gradient_loss(descriminator, real, fake)
    desc_loss_list.append(loss.item())
    penalty_list.append(gradient_penalty.item())
    loss = loss + gradient_penalty_coeff*gradient_penalty
    return loss


def descriminator_part(descriminator, generator, real, z_size, \
                 desc_loss_list, penalty_list, label,\
                 gradient_penalty_coeff=0.1):
    device = label.device
    #descriminator true part
    real.requires_grad_(True)
    descr_real = descriminator(real, label)
    #descriminator fake part
    with torch.no_grad():
        z = torch.randn(real.shape[0], z_size)
        z = z.to(device)
        fake_label = torch.randint_like(label,2)
        fake = generator(z, fake_label)
    fake.requires_grad_(True)
    descr_fake = descriminator(fake, fake_label)
    loss = discriminator_logistic_non_saturating(descr_fake, descr_real)
    gradient_penalty = zero_centered_gradient_loss(descr_real, descr_fake, real, fake)
    desc_loss_list.append(loss.item())
    penalty_list.append(gradient_penalty.item())
    loss = loss + gradient_penalty_coeff*gradient_penalty
    return loss

#for human genome frequences are
#'A':0.2733 'C':0.2296 'T':0.2284 'G':0.2685
def mini_batch_descrimination(batch, temp=15.0, \
    weights=[0.2733, 0.2296, 0.2284, 0.2685]):
    weights = 1 / torch.FloatTensor(weights)
    weights = weights / torch.sum(weights)
    weights = weights.to(batch.device)
    div_matrix = torch.zeros(batch.shape[0], batch.shape[0])
    div_matrix = div_matrix.to(batch.device)
    for i in range(batch.shape[0] - 1):
        for j in range(i+1, batch.shape[0]):
            vector1 = batch[i,:,:]
            vector1_norm = torch.norm(vector1, dim=0)
            vector1_norm = vector1_norm[None,:].repeat(batch.shape[1],1)
            vector1 = vector1 / vector1_norm
            with torch.no_grad():
                dna1 = torch.argmax(vector1, dim=0)
                mol_weights1 = weights[dna1]
                max_weights1 = torch.sqrt(vector1.max(dim=0)[0])
                mol_weights1 = mol_weights1 / max_weights1
            vector2 = batch[j,:,:]
            with torch.no_grad():
                dna2 = torch.argmax(vector2, dim=0)
                mol_weights2 = weights[dna2]
                max_weights2 = torch.sqrt(vector2.max(dim=0)[0])
                mol_weights2 = mol_weights2 / max_weights2
            vector2_norm = torch.norm(vector2, dim=0)
            vector2_norm = vector2_norm[None,:].repeat(batch.shape[1],1)
            vector2 = vector2 / vector2_norm
            inner_prod_vect = vector1*vector2
            inner_prod_sum = inner_prod_vect.sum(dim=0)
            inner_prod_weight = mol_weights1 * mol_weights2 * inner_prod_sum
            inner_prod = inner_prod_weight.mean()
            div_matrix[i,j] = torch.exp(temp*(inner_prod))
    loss = div_matrix[i,j].mean()
    return loss


def frequence_loss(batch, frequences=[0.2733, 0.2296, 0.2284, 0.2685]):
    frequences_estim = batch.mean(dim=0)
    frequences_estim = frequences_estim.mean(dim=1)
    loss_f = torch.nn.MSELoss()
    with torch.no_grad():
        frequences = torch.FloatTensor(frequences).to(batch.device)
    loss = 10*loss_f(frequences_estim, frequences)
    return loss 


def generator_part(descriminator, generator, input, \
                  z_size, gen_loss_list, batch_diver_list, \
                  frequency_loss_list, label):
    #generator part
    device = label.device
    z = torch.randn(input.shape[0], z_size)
    z = z.to(device)
    fake_label = torch.randint_like(label,2)
    fake = generator(z, fake_label)
    generator_fake = descriminator(fake, fake_label)
    loss = generator_logistic_non_saturating(generator_fake)
    batch_diversity = mini_batch_descrimination(fake)
    freq_loss = frequence_loss(fake)
    batch_diver_list.append(batch_diversity.item())
    gen_loss_list.append(loss.item())
    frequency_loss_list.append(freq_loss.item())

    return  loss #+batch_diversity


def generator_part_wasser(descriminator, generator, input, \
                  z_size, gen_loss_list, batch_diver_list, label):
    device = label.device
    #generator part
    z = torch.randn(input.shape[0], z_size)
    #z[z>2] = 2 #tales trancetion
    z = z.to(device)
    fake = generator(z, label)
    generator_fake = descriminator(fake)
    loss = -generator_fake.mean()
    batch_diversity = mini_batch_descrimination(fake)
    batch_diver_list.append(batch_diversity.item())
    gen_loss_list.append(loss.item())
    return  loss


def end_of_gan_epoch(generator, descriminator, input, z_size, label, path_to_save=None):
        z = torch.randn(input.shape[0], z_size)
        z = z.to(label.device)
        generator.eval()
        fake = generator(z, label)
        fake = fake.detach().cpu()
        print(fake[0,:,99].tolist())
        print(fake[0,:,200].tolist())
        print(fake[0,:,313].tolist())
        
        for k in range(4):
            dna = onehot_to_str(fake[k,:,:].numpy().transpose())
            print(dna)

        mol_stats = generator_stats(generator, z_size, label, 1)
        print("mm kouz:  'A':0.2733 'C':0.2296 'T':0.2284 'G':0.2685")
        print("gen:", mol_stats)
        generator_4mers_dict = generator_stats(generator, z_size, label, 4)
        with open("weights/kouzine_mm_stat.json", "r") as outfile:
            hh_kmer_dict = json.load(outfile)

        generator.train()

        delta_p = [generator_4mers_dict[key] - hh_kmer_dict[key] for key in hh_kmer_dict]
        delta_p = np.abs(delta_p).sum() / 2
        print("4mers discrepency:", delta_p)

        if delta_p < 0.095:
            if path_to_save is not None:
                path_to_save_g = os.path.join(path_to_save, "generator.pth")
                torch.save(generator.state_dict(), path_to_save_g)
                path_to_save_d = os.path.join(path_to_save, "descriminator.pth")
                torch.save(descriminator.state_dict(), path_to_save_d)
