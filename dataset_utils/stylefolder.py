import torch
import torchvision
import torchvision.datasets as datasets
from torchvision.datasets.folder import IMG_EXTENSIONS
import torchvision.transforms as transforms
from typing import Optional, Any, Tuple, Callable
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import os
import opencc


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")

class StyleFolder(datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root=root,
            loader=loader,  # default pil_loader
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,  # default None
            is_valid_file=is_valid_file,
        )
        self.transform = transform
        # self.imgs = self.samples # samples (list): List of (sample path, class_index) tuples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, original)
            sample noisy image + standard font
            original original image.
        """
        path, target = self.samples[
            index
        ]  # samples (list): List of (sample path, class_index) tuples
        character = os.path.basename(os.path.dirname(path))
        styled = self.loader(path)  # styled image
        standard = char2img(character, cvt_traditional=False)
        standard = self.transform(standard)
        styled = self.transform(styled)


        # torch.squeeze(img2tensor(original))  # torch.squeeze((255 * img2tensor(original)).to(torch.uint8))

        return standard, styled

    def __len__(self) -> int:
        return len(self.samples)

def build_dataset_cgan(args):
    root = args.data_path
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
    ])
    dataset = StyleFolder(root, transform=transform)

    return dataset

def char2img(character: str, font=None, cvt_traditional=False):
    assert len(character) == 1
    if not font:
        font = ImageFont.truetype('FZ-Std-Kai.ttf', size=200)
    if cvt_traditional:
        character = opencc.OpenCC.convert(character)
    image = Image.new('RGB', (224, 224), color='black')
    draw = ImageDraw.Draw(image)

    # text_size = draw.textsize(character, font=font)
    text_location = (image.width // 2, image.height // 2)
    draw.text(text_location, character, font=font, fill='white', anchor='mm')
    # gray_array = np.asarray(image.convert('L'))

    return image.convert('L')


def main(args):
    print(args)
    dataset = build_dataset_cgan(args=args)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=9)

    for batch_data in data_loader:
        # Extract inputs and labels from the batch
        standard, styled = batch_data
        break
    print(f'styled {styled.shape}')
    print(f"styled max {torch.max(styled)}")
    print(f'standard {standard.shape}')
    print(f"standard max {torch.max(standard)}")

    grid_img = torchvision.utils.make_grid(standard, nrow=3)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title('standard')
    plt.show()

    grid_img = torchvision.utils.make_grid(styled, nrow=3)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title('styled')
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "style dataset display")
    parser = argparse.ArgumentParser("Style dataset module", add_help=False)
    parser.add_argument("--data_set", default="image_folder", type=str)
    parser.add_argument("--data_path", default="../../Data/DicData", type=str)
    parser.add_argument("--image_size", default=64, type=int)
    args = parser.parse_args()
    main(args)