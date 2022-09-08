# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
# Captioning network for BycycleGAN (https://github.com/junyanz/BicycleGAN) and
# Modified from Ligdream (https://github.com/compsciencelab/ligdream)
# Modification of the original code to use libmolgrid for input preparation 8/04/22

import dataloader
import baseoptions
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from networks import EncoderCNN_v3, DecoderRNN, VAE
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
import sys
import numpy as np

args = baseoptions.BaseOptions().create_parser()

# ---------
# Input output paths
# ----------

smiles_path = args['input']
save_models = args['output_dir']
hdf5_path = '/c7/scratch2/Mariana/cnns/datacaptionnetwork/MolPort_ligs/Prep/capNWdatav3.hdf5'

# --------
# initialize tensorboard
# ---------

tb = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------
#  Networks, loss functions and optimizers
# ---------

# VAE
vae_model = VAE(15).to(device)
vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=args['lr_VAE'])

checkpoint = torch.load('weights/VAE_5-35000.pth')
vae_model.load_state_dict(checkpoint['model_state_dict'])
vae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
vae_model.train()

# caption
encoder = EncoderCNN_v3(15).to(device)
decoder = DecoderRNN(512, 1024, 29, 1).to(device) #  input_dim=512, hidden_dim=1024, length_vocab=29, num_layers=1

criterion = nn.CrossEntropyLoss()
caption_params = list(decoder.parameters()) + list(encoder.parameters())
caption_optimizer = torch.optim.Adam(caption_params, lr=args['lr_cap'])

encoder.train()
decoder.train()

models = [encoder, decoder, vae_model]

def cross_entropy(real_tensor, fake_tensor, eps=1e-5):
    CE = -1*((real_tensor * torch.log(fake_tensor + eps)).sum(dim=1).mean())
    return CE

def KLD(mu, logvar):
    # mean or sum?
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    print('KLD alone %f' % KLD)
    return KLD

def channel_pearson_corr(output_tensor,target_tensor):
    # mean by channel and reshape to do substraction
    voutput_tensor = output_tensor - output_tensor.mean(dim = [2,3,4]) \
        .view(output_tensor.shape[0], output_tensor.shape[1], 1, 1, 1)
    vtarget_tensor = target_tensor - target_tensor.mean(dim = [2,3,4]) \
        .view(target_tensor.shape[0], target_tensor.shape[1], 1, 1, 1)
    # pearson correlation between each channels but for all elements in the batch. cost.shape would be = [14]
    pc = torch.sum((voutput_tensor * vtarget_tensor),
                   dim = [0,2,3,4]) / (torch.sqrt(torch.sum((voutput_tensor ** 2),
                                                            dim = [0,2,3,4])) * torch.sqrt(torch.sum((vtarget_tensor ** 2),
                                                                                                     dim = [0,2,3,4])))
    # pearson correlation between each channels for each element in batch change dim to [2,3,4]. cost.shape would be = [batch_size,14]
    return pc

# ----------
#  Start data loader
# ----------

batch_size = args['batch_size']

dataset = dataloader.CustomDataset(smiles_path, hdf5_path)
dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = args['num_workers'])

number_ligs = len(dataloader.dataset)

# Normalize tensor values
def normVoid_tensor(input_tensor):
    # normalize?? this does not add to 1
    input_tensor /= input_tensor.sum(axis = 1).max()
    # add void dimension
    input_tensor = torch.cat((input_tensor, (1 - input_tensor.sum(dim = 1).unsqueeze(1))), axis=1)
    # clamp values
    input_tensor = torch.clamp(input_tensor, min=0, max=1)
    # normalize again adds to one
    input_tensor /= input_tensor.sum(axis = 1).unsqueeze(1)
    return input_tensor

# ----------
#  Training
# ----------

# start to train caption network after iteration=caption_start
caption_start = args['start_cap']
epochs = args['epochs']
number_iters = (number_ligs // batch_size) * int(epochs)

# weight KLD
number_weights = (number_ligs // batch_size) * caption_start
x = np.linspace(-10, 10, number_weights)
B_values = 1/(1 + np.exp(-x))

#k = 0
k = 24891
prev_time = time.time()
for epoch in range(4, epochs):
    for dinput_tensor, caption, lengths in dataloader:

        # lengths must be a list
        lengths = lengths.tolist()

        # normalize density values
        dinput_tensor = normVoid_tensor(dinput_tensor).to(device)

        vae_optimizer.zero_grad()
        recon_batch, mu, logvar = vae_model(dinput_tensor)
        CE = cross_entropy(dinput_tensor, recon_batch)
        B_i = (k if k < number_weights else (number_weights-1))
        _KLD = KLD(mu, logvar) * B_values[B_i]
        vae_loss = (CE + _KLD ).to(device)

        print('vale_loss %f CE %f KLD %f b %f' % (vae_loss.item(), CE.item(), _KLD.item(), B_values[B_i]))

        vae_loss.backward()
        vae_optimizer.step()
        recon_batch = recon_batch.detach()

        cPC = channel_pearson_corr(dinput_tensor, recon_batch)
        print('PCSTART %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f'%(cPC[0], cPC[1], cPC[2],
                                                                      cPC[3], cPC[4], cPC[5],
                                                                      cPC[6], cPC[7], cPC[8],
                                                                      cPC[9], cPC[10], cPC[11],
                                                                      cPC[12], cPC[13], cPC[14]))

        if k % 1000 == 0:
            torch.save(dinput_tensor, "shapes/ori_lig_%d-%d.pt" % (epoch,k))
            torch.save(recon_batch, "shapes/vae_lig_%d-%d.pt" % (epoch, k))

        # tensorboard to visualize the loss functions
        tb.add_scalar("VAE Loss", vae_loss.item(), k)
        tb.add_scalar("CE recons", CE.item(), k)

        # count the time left
        iters_left = number_iters - k
        time_left = datetime.timedelta(seconds = iters_left * (time.time()- prev_time))
        prev_time = time.time()

        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] ETA: %s \n" %(epoch, epochs, k, number_iters, time_left))

        if k == 10000:
            for g in vae_optimizer.param_groups:
                print('old VAE lr ', str(g['lr']))
                lr = g['lr'] / 2.
                g['lr'] = lr
                print('new VAE lr', str(g['lr']))

        if epoch >= caption_start:

            captions = caption.to(device)
            # only take tensor with values
            targets = pack_padded_sequence(captions,
                                           lengths,
                                           batch_first=True,
                                           enforce_sorted= False)[0] #shape = sum_length_sequences in batch

            decoder.zero_grad()
            encoder.zero_grad()

            features = encoder(recon_batch)
            outputs = decoder(features, captions, lengths) # shape = sum_length_sequences in batch x vocab_length
            cap_loss = criterion(outputs, targets)

            cap_loss.backward()
            caption_optimizer.step()
            itcap_loss = cap_loss.item()

            # tensorboard to visualize the loss functions
            tb.add_scalar("Caption Loss", itcap_loss, k)

            print('CaptionBCE: %f' % itcap_loss)

            # Reduce the LR
            if k % 100000 :
                for param_group in caption_optimizer.param_groups:
                    print('old capt lr ', str(param_group['lr']))
                    lr = param_group["lr"] / 2.
                    param_group["lr"] = lr
                    print('new capt lr ', str(param_group['lr']))

        # save networks every N iterations
        if k % 1000 == 0:
            torch.save({'epoch':k,
                        'model_state_dict':vae_model.state_dict(),
                        'optimizer_state_dict':vae_optimizer.state_dict()},
                       "weights/VAE_%d-%d.pth" % (epoch, k))
            if epoch == caption_start:
                torch.save({'epoch':k,
                            'model_state_dict':encoder.state_dict(),
                            'optimizer_state_dict':caption_optimizer.state_dict()},
                           "weights/encoder_%d-%d.pth" % (epoch, k))
                torch.save({'epoch':k,
                            'model_state_dict':decoder.state_dict(),
                            'optimizer_state_dict':caption_optimizer.state_dict()},
                           "weights/decoder_%d-%d.pth" % (epoch,k))

        k += 1

