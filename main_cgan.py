import argparse
import torch
from torch import nn
from torch import optim
import os

from dcgan import Discriminator
from cgan import AeGenerator, UnetGenerator
from utils import read_yaml_config, overwrite_config, \
    get_args_parser, save_model, auto_load_model, \
    setup_device, TensorboardLogger
from dataset_utils.stylefolder import build_dataset_cgan
from engine import train_one_epoch_cgan, CGanEvalLogger

def main(args):
    device = setup_device()
    torch.manual_seed(args.seed)

    dataset = build_dataset_cgan(args=args)

    data_loader = torch.utils.data.DataLoader(
    dataset, shuffle=True,
    batch_size=args.batch_size,
    pin_memory=args.pin_mem)

    gen_model = UnetGenerator(args.latent_size, args.n_noise, args.feature_maps_g, args.input_channel)
    dis_model = Discriminator(args.feature_maps_d, args.input_channel * 2)  # discriminator takes both base and styled images
    gen_model.to(device)
    dis_model.to(device)

    g_optim = optim.Adam(gen_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_optim = optim.Adam(dis_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    start_epoch = 0

    if args.auto_resume:
        auto_load_model(args, model_name="generator", model=gen_model, optimizer=g_optim)
        start_epoch = auto_load_model(args, model_name="discriminator", model=dis_model, optimizer=d_optim) + 1

    args.epochs += start_epoch

    gen_eval_logger = CGanEvalLogger(device, args.n_noise, img_num=64, dir="./gen_eval", dump=True, dataset=dataset)
    metrics_logger = TensorboardLogger(log_dir=args.log_dir)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch_cgan(generator=gen_model, discriminator=dis_model,
                        l1_weight=args.l1_weight, g_optim=g_optim, d_optim=d_optim,
                        dataloader=data_loader, device=device, epoch=epoch,
                        metric_logger=metrics_logger, print_freq=args.print_freq, 
                        gen_eval_logger=gen_eval_logger, gen_eval_freq=args.gen_eval_freq)
        save_model(args, model_name="generator", epoch=epoch, model=gen_model, optimizer=g_optim)
        save_model(args, model_name="discriminator", epoch=epoch, model=dis_model, optimizer=d_optim)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "DCGAN script for Zi (Chinese hand writing) generation", parents=[get_args_parser()])
    args = parser.parse_args()
    config = read_yaml_config(args.config)
    args = overwrite_config(config, args)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)   