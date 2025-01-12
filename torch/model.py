import fnmatch
import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

INFERENCE_WIDTH = 320
INFERENCE_HEIGHT = 180
INFERENCE_SIZE = (3, INFERENCE_HEIGHT, INFERENCE_WIDTH)


class MeanStdNormalization(nn.Module):

    def __init__(self):
        super(MeanStdNormalization, self).__init__()

    def forward(self, x, mask):
        return (x - torch.mean(x)) / torch.std(x), mask


inputNormalization = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.Normalize([94.9309, 95.9795, 77.9645], [53.8035, 54.9085, 60.3234]),
    #    MeanStdNormalization()
])


@dataclass(frozen=True)
class ModelConfig(ABC):

    @property
    @abstractmethod
    def class_name(self) -> str:
        pass

    @property
    def name(self) -> str:
        return f"{self.class_name}-{self.key}-" if self.key else f"{self.class_name}-"

    key: str = ""
    description: str = ""


@dataclass(frozen=True)
class Metrics:
    train_losses: List[float]
    valid_losses: List[float]
    valid_accuracy: List[float]
    accuracy: List[float]


def from_dict(cls, data):
    field_types = {
        field.name: field.type for field in cls.__dataclass_fields__.values()}
    # Build the class by converting fields recursively
    return cls(**{
        key: from_dict(field_types[key], value) if isinstance(value, dict) and hasattr(
            field_types[key], "__dataclass_fields__") else value
        for key, value in data.items()
    })


def from_config(config_dict):
    class_name = config_dict['class_name']
    return from_dict(globals()[class_name].Config, config_dict)


def from_metrics(metrics_dict) -> Optional[Metrics]:
    return None if metrics_dict is None else from_dict(Metrics, metrics_dict)


class Catalog:

    @dataclass(frozen=True)
    class Entry:
        model: Optional[nn.Module]
        config: ModelConfig
        metrics: Optional[Metrics]

    Singleton: Optional['Catalog'] = None

    directory: str
    catalog_filename: str
    catalog: Dict[str, 'Catalog.Entry']

    def __init__(self, directory: str, catalog_filename: str = 'catalog.json'):
        self.directory = directory
        self.catalog_filename = os.path.join(self.directory, catalog_filename)
        os.makedirs(self.directory, exist_ok=True)
        self.load_catalog(self.catalog_filename)

    @staticmethod
    def get(directory: str = 'models') -> 'Catalog':
        if Catalog.Singleton is None:
            Catalog.Singleton = Catalog(directory, 'catalog.json')
        return Catalog.Singleton

    def load_catalog(self, catalog_filename: str):
        if os.path.exists(catalog_filename):
            with open(catalog_filename, 'r') as fp:
                catalog = json.load(fp)
            self.catalog = {
                key: Catalog.Entry(
                    model=None,
                    config=from_config(entry['config']),
                    metrics=from_metrics(entry['metrics'])
                ) for key, entry in catalog.items()}
        else:
            self.catalog = {}

    def save_catalog(self):
        with open(self.catalog_filename, 'w+') as fp:
            json.dump({key: asdict(replace(entry, model=None)) for key, entry in self.catalog.items()},
                      fp, indent=4)

    def save(self, model: nn.Module, config: ModelConfig):
        entry = self.catalog.get(config.name)
        if entry is None:
            self.catalog[config.name] = Catalog.Entry(model, config, None)
            self.save_catalog()
        elif entry.model is not None:
            print(f"Model {config.name} already created, using existing.")
            model = entry.model
        else:
            self.catalog[config.name] = Catalog.Entry(model, config, None)
        return model

    def save_state(self, model: nn.Module, metrics: Metrics):
        # Save the states in its own file.
        state_filename = os.path.join(
            self.directory, f"{model.config.name}.pth")
        torch.save(model.state_dict(), state_filename)
        # Save the metrics to the catalog:
        name = model.config.name
        entry = self.catalog[name]
        assert entry is not None
        self.catalog[name] = replace(entry, metrics=metrics)

    def load(self, name: str) -> Optional[nn.Module]:
        entry = self.catalog[name]
        if entry is None:
            return None
        model = entry.model
        if model is None:
            model = self.create(entry.config)
            filename = os.path.join(
                self.directory, f"{entry.config.name}.pth")
            if os.path.exists(filename):
                model.load_state_dict(
                    torch.load(filename, weights_only=True)
                )
            self.catalog[name] = Catalog.Entry(
                model, entry.config, entry.metrics)
        return model

    def create(self, config: ModelConfig) -> nn.Module:
        cls = globals()[config.class_name]
        model = cls(config)
        model = self.save(model, config)
        return model

    def list(self, patterns: Optional[Union[str, List[str]]] = None) -> List[str]:
        values = [entry.config.name for entry in self.catalog.values()]
        if not patterns:
            return values
        if isinstance(patterns, str):
            patterns = [patterns]

        def match(s: str) -> bool:
            return any(fnmatch.fnmatchcase(s, p) for p in patterns)
        return [s for s in values if match(s)]

    def metrics(self, name: str) -> Optional[Metrics]:
        entry = self.catalog[name]
        return None if entry is None else entry.metrics


def height_matrix(height: int, width: int) -> torch.Tensor:
    col = np.arange(height)
    mat = np.zeros((height, width, 1), dtype=np.float32)
    for r in range(0, width):
        mat[:, r, 0] = height - col
    mat = (mat - np.mean(mat)) / np.std(mat)
    return torch.tensor(mat.reshape((1, height, width, 1))).permute(0, 3, 1, 2)


def width_matrix(height: int, width: int) -> torch.Tensor:
    row = np.arange(width)
    mat = np.zeros((height, width, 1), dtype=np.float32)
    for c in range(0, height):
        mat[c, :, 0] = height - row
    mat = (mat - np.mean(mat)) / np.std(mat)
    return torch.tensor(mat.reshape((1, height, width, 1))).permute(0, 3, 1, 2)


class PositionalLayer(nn.Module):
    # Number of channels added for position anchoring.
    channel_count = 2

    x_pos: nn.Parameter
    y_pos: nn.Parameter

    def __init__(self):
        super(PositionalLayer, self).__init__()
        # Learnable x-coordinates
        self.x_pos = nn.Parameter(torch.randn(
            1, 1, INFERENCE_HEIGHT, INFERENCE_WIDTH))
        # Learnable y-coordinates
        self.y_pos = nn.Parameter(torch.randn(
            1, 1, INFERENCE_HEIGHT, INFERENCE_WIDTH))

    def forward(self, x):
        batch_size = x.shape[0]

        # Concatenate positional embeddings with input
        x = torch.cat([
            x,
            self.x_pos.expand(batch_size, -1, -1, -1),
            self.y_pos.expand(batch_size, -1, -1, -1)
        ], dim=1)

        return x


class BasicModel(nn.Module):

    @dataclass(frozen=True)
    class Config(ModelConfig):
        class_name: str = "BasicModel"
        kernel_size: Tuple[int, int] = (9, 9)
        dropout_rate: float = 0.1
        output_channels: List[int] = field(
            default_factory=lambda: [32, 16, 8, 4])
        decoder: Literal["conv", "linear"] = "linear"

        @property
        def name(self) -> str:
            return (
                super().name +
                f"k{'x'.join(map(str, list(self.kernel_size)))}"
                f"-f{':'.join(map(str, self.output_channels))}"
                f"-{self.decoder}"
            )

    config: Config

    @property
    def name(self) -> str:
        return self.config.name

    sequential: nn.Sequential
    positional: PositionalLayer
    decoder: Union[nn.Linear, nn.Conv2d]

    def __init__(self, config: Optional[Config] = None):
        super(BasicModel, self).__init__()
        self.config = config or self.Config()
        # Extracts parameters from the config dict:
        kernel_size = self.config.kernel_size
        dropout_rate = self.config.dropout_rate
        channels = [3 + PositionalLayer.channel_count] + \
            self.config.output_channels
        # Creates the network:
        self.positional = PositionalLayer()
        layers = OrderedDict()
        for idx, (inch, outch) in enumerate(zip(channels[:-1], channels[1:])):
            layers[f"conv{idx}"] = nn.Conv2d(
                inch, outch, kernel_size, padding='same')
            layers[f"relu{idx}"] = nn.ReLU()
            layers[f"dropout{idx}"] = nn.Dropout(dropout_rate)
        self.sequential = nn.Sequential(layers)
        if self.config.decoder == "conv":
            self.decoder = nn.Conv2d(
                channels[-1],
                1, kernel_size, padding='same')
        else:
            self.decoder = nn.Linear(
                channels[-1], 1)

    def forward(self, x):
        x = self.positional(x)
        x = self.sequential(x)
        if self.config.decoder == "conv":
            x = self.decoder(x)
            x = x.permute(0, 2, 3, 1)
        else:
            # (B, F, H, W) -> (B, H, W, F)
            x = x.permute(0, 2, 3, 1)
            x = self.decoder(x)
        return F.sigmoid(x).squeeze(3)

    def to(self, *args, **kwargs):
        device = args[0] if args is not None and len(args) > 0 else None
        if isinstance(device, (torch.device, str)):
            self.positional = self.positional.to(device)
        return super().to(*args, **kwargs)


class DilatedLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
        super(DilatedLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=1, padding='same', dilation=dilation)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        return self.dropout(F.relu(self.conv(x)))


class BanksModel(nn.Module):

    @dataclass(frozen=True)
    class Config(ModelConfig):
        class_name: str = "BanksModel"
        kernel_size: Tuple[int, int] = (5, 5)
        encoder_channels: int = 16
        decoder_channels: int = 16
        dropout_rate: float = 0.1

        @property
        def name(self) -> str:
            return (
                super().name +
                f"k{'x'.join(map(str, list(self.kernel_size)))}" +
                f"-codec{self.encoder_channels}:{self.decoder_channels}"
            )

    config: Config

    @property
    def name(self) -> str:
        return self.config.name

    def __init__(self, config: Optional[Config] = None):
        super(BanksModel, self).__init__()
        self.config = config or self.Config()
        # Extracts parameters from config:
        kernel_size = self.config.kernel_size
        encoder_channels = self.config.encoder_channels
        decoder_channels = self.config.decoder_channels
        dropout_rate = self.config.dropout_rate
        channel_count = 3 + PositionalLayer.channel_count
        # Creates the network:
        self.positional = PositionalLayer()
        self.inputs1 = DilatedLayer(
            channel_count, encoder_channels, kernel_size, 2, dropout_rate)
        self.inputs2 = DilatedLayer(
            channel_count, encoder_channels, kernel_size, 2, dropout_rate)
        self.inputs3 = DilatedLayer(
            channel_count, encoder_channels, kernel_size, 3, dropout_rate)
        self.inputs4 = DilatedLayer(
            channel_count, encoder_channels, kernel_size, 4, dropout_rate)
        self.deconv = nn.Conv2d(4 * encoder_channels,
                                decoder_channels, kernel_size, padding='same')
        self.dropout = nn.Dropout2d(dropout_rate)

        self.fc = nn.Linear(decoder_channels, 1)

    def forward(self, x):
        x = self.positional(x)
        x1 = self.inputs1(x)
        x2 = self.inputs2(x)
        x3 = self.inputs3(x)
        x4 = self.inputs4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.dropout(F.relu(self.deconv(x)))
        x = x.permute(0, 2, 3, 1)              # (B, F, W, H) -> (B, W, H, F)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x.squeeze(3)

    def to(self, *args, **kwargs):
        device = args[0] if args is not None and len(args) > 0 else None
        if isinstance(device, (torch.device, str)):
            self.positional = self.positional.to(device)
        return super().to(*args, **kwargs)
