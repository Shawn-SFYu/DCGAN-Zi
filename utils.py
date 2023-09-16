import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
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
        add_help=False
    )
    parser.add_argument("-c", "--config", required=True, help="path to yaml config")
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--data_path", default="../Data/DicData", type=int, help="latent size for GAN")
    parser.add_argument("--latent_size", default=None, type=int, help="latent size for GAN")
    parser.add_argument("--input_channel", default=None, type=int, help="input image channel number")
    parser.add_argument("--image_size", default=None, type=int, help="input image size")
    parser.add_argument("--beta1", default=0.5, type=float, help="beta1 in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta2 in Adam")
    parser.add_argument("--feature_maps_g", default=None, type=int, help="beta2 in Adam")
    parser.add_argument("--feature_maps_d", default=None, type=int, help="beta2 in Adam")
    parser.add_argument("--pin_mem", default=True, type=bool, help="pin memory")
    parser.add_argument(
    "--lr",
    type=float,
    default=0.0002,
    metavar="LR",
    help="learning rate")

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
        self.writer = SummaryWriter(logdir=log_dir)
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