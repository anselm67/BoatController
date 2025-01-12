#!/usr/bin/env python3

import glob
import os
import random
from typing import List, Optional, Tuple, Union

import click
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.tv_tensors import Mask

from model import INFERENCE_HEIGHT, INFERENCE_WIDTH, inputNormalization
from utils import img_name
from viewer import Viewer


def rgb2tag(imgtag: torch.Tensor) -> torch.Tensor:
    # Defauts to water
    tag = torch.zeros((imgtag.shape[1], imgtag.shape[2]), dtype=torch.float32)
    tag[imgtag[1, :, :] == 128] = 1  # Banks
    tag[imgtag[0, :, :] == 128] = 1  # Sky -> Banks
    return tag


class BanksDataset(Dataset[Tuple[torch.Tensor, torch.Tensor()]]):

    tagfiles: List[str]
    imgfiles: List[str]
    transforms: Optional[v2.Compose]

    def __init__(self,
                 tagfiles: List[str],
                 transforms: Optional[v2.Compose]):
        super(BanksDataset, self).__init__()
        self.tagfiles = tagfiles
        self.imgfiles = [img_name(t) for t in tagfiles]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.tagfiles)

    def __getitem__(self, idx: int):
        img = read_image(self.imgfiles[idx])
        tag = read_image(self.tagfiles[idx])
        tag = Mask(tag)

        if self.transforms is not None:
            img, tag = self.transforms(img, tag)

        return img.to(torch.float32), rgb2tag(tag)


def split_dataset(datadir: str, split: float = 0.2) -> Tuple[List[str], List[str]]:
    tagfiles = glob.glob(os.path.join(datadir, 'tags.*.png'))
    random.shuffle(tagfiles)
    valid_size = int(split * len(tagfiles))
    return tagfiles[valid_size:], tagfiles[:valid_size]


def create_dataset(tagfiles: Union[str, List[str]],
                   transforms: Optional[v2.Compose] = None
                   ) -> BanksDataset:
    if isinstance(tagfiles, str):
        tagfiles = glob.glob(os.path.join(tagfiles, 'tags.*.png'))
    return BanksDataset(tagfiles, transforms)


def check_images(datadir: str):
    tagfiles = glob.glob(os.path.join(datadir, 'tags.*.png'))
    for tagfile in tagfiles:
        tags = read_image(tagfile)
        if tags.shape != (3, 360, 640):
            print(f"{tagfile}: {tags.shape}")
    for imgfile in [img_name(t) for t in tagfiles]:
        img = read_image(imgfile)
        if img.shape != (3, 360, 640):
            print(f"{imgfile}: {img.shape}")


@click.command()
@click.argument("source", nargs=-1,
                type=click.Path(exists=True, dir_okay=True, readable=True),
                required=True)
@click.option("--hide-tags", is_flag=True,
              help="Hide labels instead of showing them top over the image.")
def preview(source: Union[str, Tuple[str]], hide_tags: bool = False):
    """
    Preview the SOURCE dataset (a directory containing .jpg and corresponding
    labels as tags.*.png.

    All transformations are applied as if the data was feeded into training.
    """
    if len(source) == 1:
        dataset = create_dataset(source[0])
    else:
        dataset = create_dataset(list(source))
    viewer = Viewer(iter(dataset))
    viewer.show_tags = not hide_tags
    viewer.show()


@click.command()
@click.argument("datadir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
                required=True)
def analyze(datadir: str):
    """
    Analyze the images in DATADIR and compute mean and std for normalization.

    These numbers must be fed into both the python normalizationm layer in model.py
    inputNormalization transformation, and in the android BoatController application 
    TorchClassifier in the infer method = each number divided by 255f as pytorch on 
    android already performs the scaling between 0 and 1.
    """
    # The tranformations here should be the same as the ones used for training,
    # expect for the normalization layer, which ever that is.
    train_ds = create_dataset(datadir, v2.Compose([
        v2.RandomRotation((- 10, 10)),
        v2.CenterCrop((int(0.8 * 360), int(0.8 * 640))),
        v2.RandomResizedCrop(
            size=(INFERENCE_HEIGHT, INFERENCE_WIDTH),
            scale=(1., 1.)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.3),
    ]))
    count, mean, std = 0, 0, 0
    for imgs, _ in DataLoader(train_ds):
        count += imgs.shape[0]
        mean += imgs.mean(dim=[0, 2, 3])
        std += imgs.std(dim=[0, 2, 3])
    print(f"mean: {mean / count} std: {std / count}")
