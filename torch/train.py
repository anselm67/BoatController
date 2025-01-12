#!/usr/bin/env python3
import random
from dataclasses import dataclass
from typing import Tuple, cast

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchinfo import summary
from torchvision.transforms import v2
from tqdm import tqdm

from dataset import create_dataset, split_dataset
from model import (
    INFERENCE_HEIGHT,
    INFERENCE_SIZE,
    INFERENCE_WIDTH,
    BasicModel,
    Catalog,
    Metrics,
    ModelConfig,
    inputNormalization,
)
from utils import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preparation
train_files, valid_files = split_dataset('/data/river/tags')

train_ds = create_dataset(train_files, v2.Compose([
    v2.RandomRotation((- 10, 10)),
    v2.CenterCrop((int(0.8 * 360), int(0.8 * 640))),
    v2.RandomResizedCrop(
        size=(INFERENCE_HEIGHT, INFERENCE_WIDTH),
        scale=(1., 1.)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.3),
    inputNormalization,
]))
valid_ds = create_dataset(valid_files, v2.Compose([
    v2.Resize((INFERENCE_HEIGHT, INFERENCE_WIDTH)),
    inputNormalization,
]))


@dataclass
class TrainState:
    config: ModelConfig
    model: nn.Module
    batch_size: int
    loss_fn: nn.BCELoss
    optimizer: optim.Adam
    scheduler: ExponentialLR
    metrics: Metrics

    @staticmethod
    def create(catalog: Catalog, config: ModelConfig, batch_size: int) -> 'TrainState':
        model = catalog.create(config).to(device)
# Compiling and setting below don't affect performance
# whatsoever, which is rather odd as RTX 4060 should be
# able to take advantage of this.
#        torch.set_float32_matmul_precision('high')
#        model = torch.compile(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001,
                               betas=(0.9, 0.999), eps=1e-8)
        return TrainState(
            config,
            model,
            batch_size,
            nn.BCELoss(),
            optimizer,
            ExponentialLR(optimizer, gamma=0.9),
            Metrics([], [], [], [])
        )

    @property
    def name(self) -> str:
        return self.config.name

    def batch(self, imgs: torch.Tensor, tags: torch.Tensor):
        self.optimizer.zero_grad()
        yhat = self.model(imgs)
        loss = self.loss_fn(yhat, tags)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def append_train_loss(self, loss):
        self.metrics.train_losses.append(loss)

    def validate(self):
        self.model.eval()
        count, total_loss, total_acc = 0, 0, 0

        with torch.no_grad():
            for imgs, tags in DataLoader(valid_ds, batch_size=self.batch_size, shuffle=True):
                imgs, tags = imgs.to(device), tags.to(device)
                count += 1
                yhat = self.model(imgs)
                loss = self.loss_fn(yhat, tags)
                total_loss += loss.item()
                total_acc += evaluate(self.model, imgs, tags)
        self.metrics.valid_losses.append(total_loss)
        self.metrics.valid_accuracy.append(total_acc / count)
        self.model.train()
        return total_loss

    def test_accuracy(self) -> float:
        self.model.eval()
        count, total_acc = 0, 0
        with torch.no_grad():
            test_ds = create_dataset('/data/river/test-tags', v2.Compose([
                v2.Resize((INFERENCE_HEIGHT, INFERENCE_WIDTH)),
                inputNormalization,
            ]))
            for imgs, tags in DataLoader(test_ds, batch_size=self.batch_size, shuffle=True):
                imgs, tags = imgs.to(device), tags.to(device)
                count += 1
                total_acc += evaluate(self.model, imgs, tags)
        self.metrics.accuracy.append(total_acc / count)
        self.model.train()
        return 100.0 * total_acc / count


@click.command
@click.option('--epochs', type=int, default=45,
              help='Number of epochs to train for.')
@click.option('--checkpoint', type=int, default=10,
              help='Number of epochs between checkpoint saves.')
@click.option('--key', type=str, default="",
              help='Key to identify model in Catalog')
@click.option('--batch-size', '-b', type=int, default=16,
              help='Training, validation and test batch size.')
@click.pass_context
def train(ctx: click.Context, epochs: int, checkpoint: int, key: str, batch_size: int):
    """
    Trains one or more models for river segmentation.

    Models are to be defined in train.py as an array of model configurations.
    """
    catalog = cast(Catalog, ctx.obj)
    # Creates the models to train.
    states = [TrainState.create(catalog, config, batch_size) for config in [
        BasicModel.Config(key=key, decoder="linear"),
        BasicModel.Config(key=key, decoder="conv"),
    ]]

    # Sets up tqdm to report progress.
    tqdm_title_width = 48
    batch_tqdm = tqdm(total=len(train_ds),
                      position=0,
                      desc="Batch progress".ljust(tqdm_title_width))
    tqdms = [tqdm(total=epochs,
                  position=1+i,
                  desc=state.config.name.ljust(tqdm_title_width),
                  bar_format="{l_bar}{bar}|{postfix}")
             for i, state in enumerate(states)]
    dataloader = DataLoader(train_ds, batch_size, shuffle=True)

    # Runs the training loop.
    for epoch in range(epochs):
        [state.model.train() for state in states]
        train_losses = [0 for _ in states]

        batch_tqdm.reset()
        for imgs, tags in dataloader:
            imgs, tags = imgs.to(device), tags.to(device)
            losses = [state.batch(imgs, tags) for state in states]
            train_losses = [tl + l for tl, l in zip(train_losses, losses)]
            batch_tqdm.update(imgs.shape[0])

        [state.scheduler.step() for state in states]

        for state, loss in zip(states, train_losses):
            state.append_train_loss(loss)

        for progress in tqdms:
            progress.update(1)
            progress.set_postfix({
                "loss": f"{loss:.4f}",
                "vloss": f"{state.validate():.4f}",
                "acc": f"{state.test_accuracy():2.2f}"
            })

        if epoch % checkpoint == 0:
            for state in states:
                catalog.save_state(state.model, state.metrics)
                catalog.save_catalog()

    # Final save before leaving training.
    for state in states:
        catalog.save_state(state.model, state.metrics)
        catalog.save_catalog()


@click.command(name='list')
@click.argument('patterns', nargs=-1)
@click.pass_context
def list_catalog(ctx: click.Context, patterns: Tuple[str]):
    """
    List all models within the catalog matching any PATTERN.
    """
    catalog = cast(Catalog, ctx.obj)
    for id in catalog.list(list(patterns)):
        print(id)


@click.command
@click.argument('patterns', nargs=-1)
@click.option('--batch-size', '-b', type=int, default=16,
              help='Batch size to feed the summary() function.')
@click.pass_context
def infos(ctx, patterns: Tuple[str], batch_size: int):
    """
    Summarize all models from the catalog matching any PATTERN.

    This will describe the model architecture and parameter sizes
    according to the provided batch size.
    """
    catalog = cast(Catalog, ctx.obj)
    for name in catalog.list(list(patterns)):
        model = catalog.load(name)
        if model is None:
            print(f"{name} not found.")
        else:
            summary(model, input_size=(batch_size,)+INFERENCE_SIZE)


@click.command
@click.argument('patterns', nargs=-1)
@click.pass_context
def compile(ctx, patterns: Tuple[str]):
    """
    Compile all models matching any PATTERN for use on mobile.

    The compiled patterns will be stored in a file named after 
    the model name and with a .ptl extension.
    """
    catalog = cast(Catalog, ctx.obj)
    for name in catalog.list(list(patterns)):
        model = catalog.load(name)
        if model is None:
            print(f"{name} not found.")
        else:
            model = model.eval()
            script = torch.jit.trace(model, torch.rand((1,) + INFERENCE_SIZE))
            script_opt = optimize_for_mobile(script)    # type: ignore
            script_opt._save_for_lite_interpreter(f"{name}.ptl")


def smooth(data, window_size=3) -> NDArray[np.float32]:
    return np.convolve(data, np.ones(window_size, dtype=np.float32) / window_size, mode='valid')


@click.command
@click.option('--which',
              type=click.Choice(['loss', 'valid_loss', 'accuracy']),
              multiple=True,
              help='Metric(s) to plot, defauts to all.')
@click.argument('patterns', nargs=-1)
@click.pass_context
def plot(ctx: click.Context, which: Tuple[str], patterns: Tuple[str]):
    """
    Plot training metrics for all models matching any PATTERN.

    Metrics are:
    - loss: for training loss,
    - valid_loss: for validation loss
    - accuracy for test dataset accuracy.
    """
    catalog = cast(Catalog, ctx.obj)
    metrics = [(name, catalog.metrics(name))
               for name in catalog.list(list(patterns))]
    if not which:
        plots = list(enumerate(('loss', 'valid_loss', 'accuracy')))
    else:
        plots = list(enumerate(which))
    fig, axes = plt.subplots(len(plots), 1)
    if len(plots) == 1:
        axes = [axes]
    for idx, plot in plots:
        for name, metric in metrics:
            assert metric is not None
            if plot == 'loss':
                title = 'Training loss'
                data = metric.train_losses
            elif plot == 'valid_loss':
                title = 'Validation losses'
                data = metric.valid_losses
            else:
                title = 'Test accuracy'
                data = smooth(metric.accuracy, window_size=5)
            axes[idx].plot(data, label=name)
        axes[idx].set_title(title)
        axes[idx].legend()
    fig.tight_layout()
    plt.show()
