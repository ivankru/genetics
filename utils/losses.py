import torch.nn.functional as F

def discriminator_logistic_non_saturating(d_result_fake, d_result_real):
    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real))
    return loss.mean()

def generator_logistic_non_saturating(d_result_fake):
    return F.softplus(-d_result_fake).mean()