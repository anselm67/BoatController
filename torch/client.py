#!/usr/bin/env python

from typing import List, Tuple, Union, cast

import click
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.io import VideoReader
from torchvision.transforms import v2

from dataset import create_dataset
from model import INFERENCE_HEIGHT, INFERENCE_WIDTH, Catalog, inputNormalization
from viewer import Viewer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eval_model(catalog: Catalog, name: str):
    model = catalog.load(name)
    assert model is not None
    model.to(device)
    model.eval()
    return model


PREDICT_TRANSFORMS = v2.Compose([
    v2.Resize((INFERENCE_HEIGHT, INFERENCE_WIDTH)),
    inputNormalization,
])

VIDEO_TRANSFORMS = v2.Compose([
    v2.Resize((INFERENCE_HEIGHT, INFERENCE_WIDTH)),
    inputNormalization,
])


def run(model: nn.Module, img: torch.Tensor, thresold: float = 0.5) -> torch.Tensor:
    outputs = model(img.unsqueeze(0).to(device)).squeeze(0)
    return (outputs > thresold)


def predict_iterator(model: nn.Module, iterator, thresold: float = 0.5):
    with torch.no_grad():
        for img, _ in iterator:
            img = img.to(device)
            yhat = model(img.unsqueeze(0).to(device)).squeeze(0)
            yield img, yhat


def video_iterator(video, flip_vertically: bool = True):
    for frame in video:
        image = VIDEO_TRANSFORMS(frame['data'])
        if flip_vertically:
            image = TF.rotate(image, angle=180),
            yield image[0], None
        else:
            yield image, None


@click.command
@click.argument('model_name', type=str, required=True)
@click.argument("source", nargs=-1,
                type=click.Path(exists=True, dir_okay=True, readable=True))
@click.option('--show-predictions', is_flag=True, type=bool, default=False,
              help='Show prediction probablities along the image and tag.')
@click.option('--flip', 'flip_vertically', is_flag=True, type=bool, default=True,
              help='Flip video images vertically')
@click.option('--thresold', type=float, default=0.5,
              help="Thresold to use for probability cut-off (between 0.0 and 1.0)")
@click.option('--pause', type=float, default=0,
              help="Pause time in between frames in milli-seconds.")
@click.pass_context
def predict(ctx,
            model_name: str,
            source: Union[str, Tuple[str]],
            thresold: float,
            pause: float,
            show_predictions: bool,
            flip_vertically: bool):
    """
    Runs the MODEL_NAME on the given image SOURCE.

    The source can be either a directory, in which case it is expected to
    have images (*.jpg) and corresponding tags (tags.*.png).
    """
    catalog = cast(Catalog, ctx.obj)
    model = load_eval_model(catalog, model_name)
    # Either a full directory or an mpeg video.
    if len(source) == 1:
        source = source[0]
        if source.endswith(".mp4"):
            video = VideoReader(source, "video")
            dataset = video_iterator(video, flip_vertically)
        else:
            dataset = create_dataset(source, PREDICT_TRANSFORMS)
    else:
        dataset = create_dataset(list(source), PREDICT_TRANSFORMS)
    viewer = Viewer(predict_iterator(model, dataset, thresold))
    viewer.show_predictions = show_predictions
    viewer.thresold = thresold
    viewer.pause = pause / 1000.0
    if viewer.show():
        return


@click.command
@click.argument('model_name', type=str, required=True)
@click.argument("datadir",
                type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.pass_context
@click.option('--thresold', type=float, default=0.5,
              help="Thresold to use for probability cut-off (between 0.0 and 1.0)")
def compare(ctx: click.Context, model_name: str, datadir: str, thresold: float):
    """
    Runs the given MODEL_NAME against the DATADIR dataset, and computes its accuracy.
    """
    catalog = cast(Catalog, ctx.obj)
    model = load_eval_model(catalog, model_name)
    dataset = create_dataset(datadir, PREDICT_TRANSFORMS)
    total_count, match_count = 0, 0
    for imgs, tags in DataLoader(dataset, batch_size=32):
        imgs, tags = imgs.to(device), tags.to(device)
        yhat = model(imgs)
        yhat = (yhat > thresold).to(torch.uint8)
        match_count += (torch.sum(yhat == tags))
        total_count += np.prod(yhat.shape)
    print(f"match: {100.0 * match_count / total_count:2.2f}%")


@click.command
@click.argument("datadir",
                type=click.Path(
                    exists=True,
                    file_okay=False,
                    dir_okay=True,
                    readable=True))
@click.argument('patterns', nargs=-1)
@click.option('--thresold', type=float, default=0.5,
              help="Thresold to use for probability cut-off (between 0.0 and 1.0)")
@click.pass_context
def rank(ctx: click.Context, datadir: str, patterns: Tuple[str], thresold: float):
    """
    Rank all models matching any PATTERN by their test accuracy.

    Each model accuracy is measured against the dataset in DATADIR. This
    command than outputs the list sorted by highest accuracy first.
    """
    catalog, _ = ctx.obj
    ranks: List[Tuple[str, float]] = []
    dataset = create_dataset(datadir, PREDICT_TRANSFORMS)
    for name in catalog.list(list(patterns)):
        model = catalog.load(name).to(device)
        model.eval()
        total_count, match_count = 0, 0
        for imgs, tags in DataLoader(dataset, batch_size=32):
            imgs, tags = imgs.to(device), tags.to(device)
            yhat = model(imgs)
            yhat = (yhat > thresold).to(torch.uint8)
            match_count += (torch.sum(yhat == tags))
            total_count += np.prod(yhat.shape)
        ranks.append((name, cast(float, 100.0 * match_count / total_count)))
    for name, accuracy in sorted(ranks, key=lambda r: - r[1]):
        print(f"{name}\t{accuracy:2.2f}")
