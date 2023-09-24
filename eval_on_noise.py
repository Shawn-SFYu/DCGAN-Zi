import argparse
import torch
from torch import optim
import torchvision.utils as vision_utils
import numpy as np

from dcgan import Discriminator
from cgan import AeGenerator, UnetGenerator
from utils import auto_load_model, setup_device, read_yaml_config, overwrite_config, get_args_parser
from dataset_utils.stylefolder import build_dataset_cgan

def main(args):
    device = setup_device()
    torch.manual_seed(args.seed)

    dataset = build_dataset_cgan(args=args)

    data_loader = torch.utils.data.DataLoader(
    dataset, shuffle=True,
    batch_size=args.eval_batch_size)

    gen_model = UnetGenerator(args.latent_size, args.n_noise, args.feature_maps_g, args.input_channel)
    dis_model = Discriminator(args.feature_maps_d, args.input_channel * 2)  # discriminator takes both base and styled images
    gen_model.to(device)
    dis_model.to(device)

    g_optim = optim.Adam(gen_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_optim = optim.Adam(dis_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))


    auto_load_model(args, model_name="generator", model=gen_model, optimizer=g_optim)
    start_epoch = auto_load_model(args, model_name="discriminator", model=dis_model, optimizer=d_optim) + 1

    input_image = next(iter(data_loader))[0][0, :, :, :]
    print(input_image.shape)
    input_image = input_image.unsqueeze(0).repeat_interleave(args.eval_batch_size, dim=0)
    print(input_image.shape)
    vision_utils.save_image(input_image, fp="input_image.png", padding=2, normalize=True)
    
    noise_vector = torch.zeros(args.eval_batch_size, args.n_noise, 1, 1)
    noise_vector[:, args.var_noise_dim, 0, 0] = torch.linspace(start=-2.0, end=2.0, steps=args.eval_batch_size)
    z, enc_dict = gen_model.encoder(input_image)
    print(z.shape)
    z = torch.concatenate((z, noise_vector), dim=1)
    print(z.shape)
    fake_img = gen_model.decoder(z, enc_dict)
    vision_utils.save_image(fake_img, fp="output_image.png", padding=2, normalize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "DCGAN script for Zi (Chinese hand writing) generation", parents=[get_args_parser()])
    parser.add_argument('--var_noise_dim', type=int, default=0, help="# of dimension set to be variable, \
                         < the total length of noise vector")
    parser.add_argument('--eval_batch_size', type=int, default=64, help="number of eval chars per image")
    args = parser.parse_args()
    config = read_yaml_config(args.config)
    args = overwrite_config(config, args)

    main(args)  