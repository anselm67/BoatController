#!/usr/bin/env python

import click
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.transforms import v2

from model import INFERENCE_HEIGHT, INFERENCE_WIDTH, inputNormalization

PREDICT_TRANSFORMS = v2.Compose([
    v2.Resize((INFERENCE_HEIGHT, INFERENCE_WIDTH)),
    inputNormalization,
])


@click.command
@click.argument('model_file', type=click.Path(exists=True, readable=True))
@click.argument('image_file', type=str, required=True)
def check(model_file: str, image_file: str):

    # Load the model
    model = torch.jit.load(model_file)
    model.eval()

    image = read_image(image_file)
    image = PREDICT_TRANSFORMS(image)

    # Run inference
    output = model(image.unsqueeze(0))
    output = (output > 0.5).squeeze(0)

    image = 0.2 * (image) + (1. - 0.2) * output

    plt.imshow(image.permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    check()
