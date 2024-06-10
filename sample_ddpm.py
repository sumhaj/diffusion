import yaml
import tqdm
import os
import argparse

import torch
import torchvision
from torchvision.utils import make_grid

from utils.noise_scheduler import NoiseScheduler
from models.unet import Unet

def sample(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    num_samples = config.num_samples

    noise_scheduler = NoiseScheduler(num_timesteps=config.diffusion.num_timesteps)
    noise_scheduler.linear_noise_scheduler(beta_start=config.diffusion.beta_start, beta_end=config.diffusion.beta_end)

    model = Unet(config.model)
    model.load_state_dict(torch.load(os.path.join(config.train.saved_ddpm_model_dir, 'min_loss.pth')))
    model.eval()

    xt = torch.randn(size=(config.num_samples, config.image_channels, config.image_size, config.image_size))
    for i in tqdm(reversed(range(config.num_timesteps))):
        noise_pred = model(xt, torch.full((num_samples, ), i))
        xt, x0_pred = noise_scheduler.sample_prev_timestep(xt, noise_pred, torch.tensor(i))
        images = torch.clamp(xt, -1., 1.).detach().cpu()
        images = (images * 127.5) + 127.5
        grid = make_grid(images, nrow=config.num_grid_rows)
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(config.diffusion.type, 'samples')):
            os.mkdir(os.path.join(config.diffusion.type, 'samples'))
        img.save(os.path.join(config.diffusion.type, 'samples', 'x0_{}.png'.format(i)))
        img.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/ddpm.yml', type=str)

    args = parser.parse_args()
    sample(args)