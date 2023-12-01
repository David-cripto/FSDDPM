from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset

PathLike = Path | str


def random_color(im: torch.Tensor) -> torch.Tensor:
    hue = 360 * np.random.rand()
    d = im * (hue % 60) / 60
    im_min, im_inc, im_dec = torch.zeros_like(im), d, im - d
    H = round(hue / 60) % 6
    cmap = [[0, 3, 2], [2, 0, 3], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]]
    return torch.cat((im, im_min, im_dec, im_inc), dim=0)[cmap[H]]


TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        random_color,
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)


def get_dataset(img_dir: PathLike) -> tuple[Dataset, ...]:
    train_dataset = torchvision.datasets.MNIST(
        img_dir, download=True, train=True, transform=TRANSFORM
    )
    test_dataset = torchvision.datasets.MNIST(
        img_dir, download=True, train=True, transform=TRANSFORM
    )
    return train_dataset, test_dataset
