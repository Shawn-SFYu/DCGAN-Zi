import os
import argparse
import torch
from torch import nn, optim
import os
from pathlib import Path
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from dataset_utils.stylefolder import StyleFolder
import yaml

def read_yaml_config(yaml_input):
    with open(yaml_input, "r") as config_file:
        config_dict = yaml.safe_load(config_file)
    config = argparse.Namespace(**config_dict)
    return config

def overwrite_config(config, args):
    for key in vars(args):
        value = getattr(args, key)
        if (value is not None) or (not hasattr(config, key)):
            setattr(config, key, value)
    return config

def get_args_parser():
    parser = argparse.ArgumentParser(
        "DCGAN for Zi Generation",
        add_help=False,
    )
    parser.add_argument("-c", "--config", required=True, help="path to yaml config")
    parser.add_argument("--epochs", type=int, help="epoch num")
    parser.add_argument("--data_path", default="../Data/DicData", type=str, help="latent size for GAN")
    parser.add_argument("--output_dir", default="./checkpoint", type=str, help="output dir")
    parser.add_argument("--latent_size", type=int, help="latent size for GAN")
    parser.add_argument("--input_channel", type=int, help="input image channel number")
    parser.add_argument("--image_size", type=int, help="input image size")
    parser.add_argument("--beta1", default=0.5, type=float, help="beta1 in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta2 in Adam")
    parser.add_argument("--feature_maps_g", type=int, help="feature_maps for generator")
    parser.add_argument("--feature_maps_d", type=int, help="feature_maps for discriminator")
    parser.add_argument("--pin_mem", default=True, type=bool, help="pin memory")
    parser.add_argument("--log_dir", default="./log_dir", type=str, help="log directory")
    parser.add_argument("--print_freq", type=int, help="print metrcis per N batch")
    parser.add_argument("--gen_eval_freq", type=bool, help="save generated images per N batch")
    parser.add_argument("--auto_resume", default=False, type=bool, help="whether load existing checkpoint")
    parser.add_argument("--save_ckpt_freq", default=1, type=int, help="ckpt saving frequency")
    parser.add_argument("--n_noise", type=int, help="length of noise vector")
    parser.add_argument("--l1_weight", type=float, help="weight of l1 term")
    parser.add_argument(
    "--lr",
    type=float,
    default=0.0002,
    metavar="LR",
    help="learning rate")

    return parser

def build_dataset(args):

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=args.data_path, transform=transform)

    return dataset


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head="scalar", step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(
                head + "/" + k, v, self.step if step is None else step
            )

    def flush(self):
        self.writer.flush()


def save_model(
    args, model_name: str, epoch: int, model: nn.Module, optimizer: optim.Optimizer
):
    output_dir = Path(args.output_dir)
    checkpoint_paths = [output_dir / (f"checkpoint-{model_name}-{epoch}.pth")]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        }

        torch.save(to_save, checkpoint_path)

    to_del = epoch - args.save_ckpt_freq
    old_ckpt = output_dir / (f"checkpoint-{model_name}-{to_del}.pth")
    if os.path.exists(old_ckpt):
        os.remove(old_ckpt)


def auto_load_model(
    args, model_name: str, model: nn.Module, optimizer: optim.Optimizer
):
    import glob
    output_dir = Path(args.output_dir)
    all_checkpoints = glob.glob(os.path.join(output_dir, f"checkpoint-{model_name}*.pth"))
    latest_ckpt = -1
    for ckpt in all_checkpoints:
        t = ckpt.split("-")[-1].split(".")[0]
        if t.isdigit():
            latest_ckpt = max(int(t), latest_ckpt)
    assert latest_ckpt >= 0, "no proper checkpoint found"
    resume_checkpoint = os.path.join(output_dir, f"checkpoint-{model_name}-{latest_ckpt}.pth")
    print("Resume checkpoint: %s" % resume_checkpoint)
    checkpoint = torch.load(resume_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"]


def setup_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        cpu_count = os.cpu_count()
        torch.set_num_threads(cpu_count-2)
    return device