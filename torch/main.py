#!/usr/bin/env python3

import random

import click
import numpy as np
import torch

from client import compare, predict, rank
from dataset import analyze, preview
from model import Catalog
from train import compile, infos, list_catalog, plot, train


def set_seed(seed: int):
    torch.manual_seed(seed)  # Seed for PyTorch
    random.seed(seed)  # Seed for Python's random module
    np.random.seed(seed)  # Seed for NumPy
    torch.cuda.manual_seed(seed)  # For CUDA, if using GPU
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    # Slows training but ensures reproducibility
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@click.group
@click.option('--catalog', '-c', 'catalog_directory',
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='models')
@click.option('--seed', type=int, default=1967,
              help='Seed for the training process if not 0.')
@click.pass_context
def cli(ctx: click.Context, catalog_directory: str, seed: int):
    print(f"Catalog {catalog_directory}, seeding with {seed}")
    if seed != 0:
        set_seed(seed)
    ctx.obj = Catalog.get(catalog_directory)


# train.py
cli.add_command(train)
cli.add_command(list_catalog)
cli.add_command(infos)
cli.add_command(compile)
cli.add_command(plot)

# cliemnt.py
cli.add_command(predict)
cli.add_command(compare)
cli.add_command(rank)

# dataset.py
cli.add_command(preview)
cli.add_command(analyze)

if __name__ == '__main__':
    cli()
