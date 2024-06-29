import yaml
import tqdm
import os
import argparse
import math

from easydict import EasyDict

from dataset.dataset import CustomDataset
from models.unet import Unet
from utils.noise_scheduler import NoiseScheduler
from utils.utils import get_optimizer

import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = EasyDict(yaml.safe_load(file))
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    print(config.dataset.mnist.train)

    mnist = CustomDataset(config.dataset.mnist.train, os.path.join(config.dataset.mnist.train, 'labels.csv'))
    mnist_loader = DataLoader(mnist, batch_size=config.train.batch_size, shuffle=True, num_workers=0)

    noise_scheduler = NoiseScheduler(num_timesteps=config.diffusion.num_timesteps)
    noise_scheduler.linear_noise_scheduler(beta_start=config.diffusion.beta_start, beta_end=config.diffusion.beta_end)

    model = Unet(config.model).to(device)
    model.train()

    optimizer = get_optimizer(config.train.optimizer, model)

    best_loss = math.inf
    for epoch_idx in range(config.train.epoch):
        losses = []
        batch = 1
        for image, _ in mnist_loader:
            optimizer.zero_grad()
            t = torch.randint(0, config.diffusion.num_timesteps, (image.shape[0], )).to(device)
            noise = torch.randn_like(image).to(device)
            noisy_image = noise_scheduler.apply_noise(image.to(device), noise, t)
            noise_pred = model(noisy_image, t)
            loss = torch.nn.SmoothL1Loss()(noise, noise_pred)
            print("Epoch : {} | Batch : {} | Loss : {}".format(epoch_idx+1, batch, loss.item()))
            batch += 1
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_loss = torch.mean(torch.tensor(losses))
        print("Epoch : {} | Loss : {}".format(epoch_idx+1, epoch_loss.item()))
        if best_loss > epoch_loss:
            torch.save(model.state_dict(), os.path.join(config.train.saved_ddpm_model_dir, 'saved_ddpm_min_loss_epoch_{}.pth'.format(epoch_idx)))
            best_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(config.train.saved_ddpm_model_dir, 'saved_ddpm_epoch_{}.pth'.format(epoch_idx)))
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='configs/ddpm.yml', type=str)

    args = parser.parse_args()
    train(args)
    

