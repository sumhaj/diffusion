import yaml
import tqdm
import os
import argparse

from easydict import EasyDict

import torch
import torchvision
from torchvision.utils import make_grid

from utils.noise_scheduler import NoiseScheduler
from models.unet import Unet

def sample(args):
    with open(args.config_path, 'r') as file:
        try:
            config = EasyDict(yaml.safe_load(file))
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    num_samples = config.train.num_samples

    noise_scheduler = NoiseScheduler(num_timesteps=config.diffusion.num_timesteps)
    noise_scheduler.linear_noise_scheduler(beta_start=config.diffusion.beta_start, beta_end=config.diffusion.beta_end)

    model = Unet(config.model)
    model.load_state_dict(torch.load(os.path.join(config.train.saved_ddpm_model_dir, 'saved_ddpm_min_loss.pth')))
    model.eval()

    xt = torch.randn(size=(num_samples, config.model.image_channels, config.model.image_size, config.model.image_size))
    for i in reversed(range(config.diffusion.num_timesteps)):
        # print('Time step : {}'.format(i))
        noise_pred = model(xt, torch.full((num_samples, ), i))
        xt, x0_pred = noise_scheduler.sample_prev_timestep(xt, noise_pred, torch.tensor(i))
    images = torch.clamp(xt, -1., 1.).detach().cpu()
    images = (images * 127.5) + 127.5
    grid = make_grid(images, nrow=config.train.num_grid_rows)
    img = torchvision.transforms.ToPILImage()(grid)
    if not os.path.exists(os.path.join(config.diffusion.type, 'samples')):
            os.mkdir(config.diffusion.type)
            os.mkdir(os.path.join(config.diffusion.type, 'samples'))
    # img.save(os.path.join(config.diffusion.type, 'samples', 'x0_{}.png'.format(i)))
    img.save(os.path.join(config.diffusion.type, 'samples', 'x0.png'))
    img.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='configs/ddpm.yml', type=str)

    args = parser.parse_args()
    sample(args)