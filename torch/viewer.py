from typing import Any, Iterator, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.image import AxesImage
from numpy.typing import NDArray


class Viewer:
    WATER = (0, 0, 255)
    BANKS = (0, 255, 0)

    iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]]
    img_display: AxesImage
    pred_display: Optional[AxesImage] = None
    mixture: float = 0.6
    pause: float = 0
    thresold: float = 0.5

    current: Optional[NDArray[np.float32]]
    terminated: bool = False
    show_predictions: bool = False
    show_tags: bool = True

    def __init__(self, iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]]):
        self.iterator = iterator

    def show(self) -> bool:
        # Gets the first image in the sequence.
        blank = np.full((360, 640, 3), 0)
        # Prepares the viewer
        if self.show_predictions:
            self.fig, axes = plt.subplots(2, 1)
            self.img_display = axes[0].imshow(blank)
            self.pred_display = axes[1].imshow(blank)
            plt.tight_layout()
        else:
            self.fig, ax = plt.subplots()
            ax.axis("off")  # Turn off axes
            self.img_display = ax.imshow(blank)
        self.fig.canvas.mpl_connect(
            "key_press_event", lambda e: self.handle(e))
        return self.loop()

    def loop(self) -> bool:
        # Enters the viewer's loop.
        while not self.terminated and self.next():

            if self.pause > 0:
                plt.pause(self.pause)
            else:
                plt.show()

        return self.terminated

    def prepare(self, img: torch.Tensor) -> NDArray[np.float32]:
        array = img.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        return np.clip(array, 0, 1)

    def pred2rgb(self, tag: torch.Tensor) -> NDArray[np.float32]:
        tag = tag.to('cpu') > self.thresold
        img = np.full(tag.shape + (3,), (255, 0, 0), dtype=np.float32)
        img[tag[:, :] == 0] = Viewer.WATER
        img[tag[:, :] == 1] = Viewer.BANKS
        return img / 255.0

    def pred2green(self, tag: torch.Tensor) -> NDArray[np.float32]:
        tag = tag.to('cpu')
        img = np.full(tag.shape + (3,), (0, 0, 0), dtype=np.float32)
        img[:, :, 0] = tag * 1.0
        img[:, :, 1] = tag * 1.0
        img[:, :, 2] = tag * 1.0
        return img

    def mix(self, img: NDArray[np.float32], tag: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.mixture * img + (1. - self.mixture) * tag

    def terminate(self):
        self.terminated = True
        plt.close('all')

    def handle(self, event: Any):
        if event.key == 'q':
            self.terminate()
            return
        elif event.key == '+':
            self.mixture = min(1.0, self.mixture + 0.1)
        elif event.key == '-':
            self.mixture = max(0, self.mixture - 0.1)
        else:
            self.next()

    def next_image(self) -> Tuple[Optional[NDArray[np.float32]], Optional[torch.Tensor]]:
        img, pred = next(self.iterator, (None, None))
        if img is None:
            plt.close()
            return None, None
        else:
            return self.prepare(img), pred

    def next(self) -> bool:
        img, pred = self.next_image()
        if img is None:
            plt.close()
            return False
        # Displays the main screen.
        if pred is not None and self.show_tags:
            self.img_display.set_data(self.mix(img, self.pred2rgb(pred)))
        else:
            self.img_display.set_data(img)
        # Display the optional pred screen
        if self.pred_display is not None and pred is not None:
            self.pred_display.set_data(self.pred2green(pred))
        self.fig.canvas.draw_idle()
        return True
