#!/usr/bin/env python
# A simple tool to tag region of an image.

import os
import sys
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from cv2.typing import MatLike


def read_image(path: str) -> MatLike:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"{path} doesn't exist.")
    if image.dtype != np.uint8:
        raise ValueError(f"{path} has unexepected type {image.dtype} .")
    else:
        return image


class Editor(object):
    GRID_COLOR = (240, 240, 240)
    WATER_COLOR = (128, 0, 0)
    BANK_COLOR = (0, 128, 0)
    SKY_COLOR = (0, 0, 128)

    TR_UP = np.array([[1, 0,  0], [0, 1, -1]], dtype=np.float32)
    TR_DOWN = np.array([[1, 0,  0], [0, 1,  1]], dtype=np.float32)
    TR_LEFT = np.array([[1, 0, -1], [0, 1,  0]], dtype=np.float32)
    TR_RIGHT = np.array([[1, 0,  1], [0, 1,  0]], dtype=np.float32)

    EDITOR_WINDOW = 'editor'
    TAG_WINDOW = 'tag'

    filenames: List[str]
    frame: MatLike
    layer: Optional[MatLike]
    prev_layer: Optional[MatLike]
    position: Tuple[int, int]

    modified: bool
    auto_save: bool
    color: Tuple[int, int, int]

    def __init__(self, filenames: List[str] | str):
        if type(filenames) is str:
            self.filenames = [filenames]
        elif type(filenames) is list:
            self.filenames = filenames
        self.layer = None
        self.show_grid = False
        self.grid_size = 25
        self.weight = 2
        self.modified = False
        self.auto_save = False
        self.color = Editor.WATER_COLOR
        self.load(0)

    def load(self, index: int):
        if index < 0 or index >= len(self.filenames):
            return False
        self.index = index
        # Loads the image and draws the grid on top.
        filename = self.filenames[self.index]
        frame = read_image(filename)
        self.filename = filename
        self.frame = frame
        self.draw_grid(frame)
        # Initializes the editor state.
        self.tagging = False
        self.modified = False
        # Loads existing layer data when available, or intializes it.
        self.prev_layer = self.layer
        layername = self.layer_filename()
        if os.path.isfile(layername):
            layer = read_image(layername)
            self.layer = cv2.resize(layer, (640, 360),
                                    interpolation=cv2.INTER_NEAREST)
            print('Loaded layer %s' % layername)
        else:
            print('Created layer %s' % layername)
            self.layer = np.zeros(frame.shape, dtype=np.uint8)

        cv2.namedWindow(Editor.EDITOR_WINDOW)
        cv2.setMouseCallback(Editor.EDITOR_WINDOW, self._mouse_callback)
        cv2.imshow(Editor.EDITOR_WINDOW, frame)
        cv2.namedWindow(Editor.TAG_WINDOW)
        cv2.setMouseCallback(Editor.TAG_WINDOW, self._mouse_callback)
        cv2.imshow(Editor.TAG_WINDOW, self.layer)   # type: ignore

    def draw_grid(self, frame: MatLike):
        if self.show_grid:
            for x in range(0, frame.shape[1], self.grid_size):
                for y in range(0, frame.shape[0], self.grid_size):
                    cv2.line(frame, (0, y),
                             (frame.shape[1], y), Editor.GRID_COLOR, 1)
                    cv2.line(frame, (x, 0),
                             (x, frame.shape[0]), Editor.GRID_COLOR, 1)

    def refresh(self):
        filename = self.filenames[self.index]
        frame = read_image(filename)
        self.frame = frame
        self.draw_grid(frame)

    def layer_filename(self) -> str:
        dirname = os.path.dirname(self.filename)
        basename, _ = os.path.splitext(os.path.basename(self.filename))
        return '%s/tags.%s.png' % (dirname, basename)

    def snap(self, coord: int) -> int:
        return self.grid_size * int(coord / self.grid_size)

    def _mix(self) -> MatLike:
        w = 0.5 + (self.weight / 10.0)
        assert self.layer is not None
        return cv2.addWeighted(self.frame, w, self.layer, 1-w, 0)

    def _mouse_callback(self, event: int, x: int, y: int, _flags: int, _params: Optional[Any]):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.tagging = True
            self.position = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.tagging = False
            self.position = (x, y)
        if self.tagging:
            x = self.snap(x)
            y = self.snap(y)
            assert self.layer is not None
            cv2.rectangle(self.layer,
                          (x, y), (x+self.grid_size, y+self.grid_size),
                          self.color, cv2.FILLED)
            cv2.imshow(Editor.EDITOR_WINDOW, self._mix())
            cv2.imshow(Editor.TAG_WINDOW, self.layer)
            self.modified = True

    def loop(self):
        while True:
            assert self.layer is not None
            cv2.imshow(Editor.EDITOR_WINDOW, self._mix())
            cv2.imshow(Editor.TAG_WINDOW, self.layer)
            key = cv2.waitKey(0) & 0xff
            if key == ord('q'):
                self.do_autosave()
                break
            if key == ord('g'):
                # Tags river banks & foliages.
                self.color = Editor.BANK_COLOR
            elif key == ord('r'):
                # Tags the sky.
                self.color = Editor.SKY_COLOR
            elif key == ord('b'):
                self.color = Editor.WATER_COLOR
            elif key == ord('f'):
                # Fills the area from current position.
                self.fill(self.position)
            elif key == ord('a'):
                self.auto_save = not self.auto_save
                print('Auto-save is {}'.format('on' if self.auto_save else 'off'))
            elif key == ord('s'):
                self.save()
            elif key == ord('n'):
                self.next()
            elif key == ord('p'):
                self.prev()
            elif key == ord('i'):
                print('Displaying grid')
                self.show_grid = not self.show_grid
                self.refresh()
            elif key == ord('w'):
                self.weight = (self.weight + 1) % 5
            elif key == ord('l'):
                if self.prev_layer is not None:
                    self.layer = self.prev_layer
                    self.modified = True
                else:
                    print('No prev layer to copy.')
            elif key == ord('h') or key == ord('?'):
                print('[B]lue/water, [R]ed/sky or [G]reen/banks.')
                print('[F]ill area, show gr[I]d, [C]lear/reset, [S]ave.')
                print('[A]uto save on / off.')
                print('[W]eight increase / decrease mixing weight (loops).')
                print('[L] copy previous layer into this layer.')
                print('[+] Increase grid size, [-] Decrease grid size.')
                print('[N]ext, [P]rev, [H]elp or [?] [Q]uit.')
            elif key == ord('c'):
                print('Resetting layer.')
                self.layer = np.full(
                    self.frame.shape, self.color, dtype=np.uint8)
                self.modified = True
            elif key == 82:
                # Translates the layer up.
                self.layer = cv2.warpAffine(
                    self.layer, Editor.TR_UP, (640, 360))
                self.modified = True
            elif key == 84:
                # Translates the layer up.
                self.layer = cv2.warpAffine(
                    self.layer, Editor.TR_DOWN, (640, 360))
                self.modified = True
            elif key == 81:
                # Translates the layer up.
                self.layer = cv2.warpAffine(
                    self.layer, Editor.TR_LEFT, (640, 360))
                self.modified = True
            elif key == 83:
                # Translates the layer up.
                self.layer = cv2.warpAffine(
                    self.layer, Editor.TR_RIGHT, (640, 360))
                self.modified = True
            elif key == ord('+'):
                if self.grid_size < 50:
                    self.grid_size += 5
                    self.refresh()
                print('Grid size: %d' % (self.grid_size))
            elif key == ord('-'):
                if self.grid_size > 5:
                    self.grid_size -= 5
                    self.refresh()
                    print('Grid size: %d' % (self.grid_size))
            else:
                print('Unknown key %d' % (key))

    def fill(self, position: Tuple[int, int]):
        x, y = position
        x = self.snap(x) + round(self.grid_size / 2)
        y = self.snap(y) + round(self.grid_size / 2)
        h, w, _ = self.frame.shape
        for dx in [-self.grid_size, 0, self.grid_size]:
            for dy in [-self.grid_size, 0, self.grid_size]:
                if x < 0 or x+dx < 0 or x >= w or x+dx >= w:
                    continue
                if y < 0 or y+dy < 0 or y >= h or y+dy >= h:
                    continue
            assert self.layer is not None
            cv2.floodFill(self.layer, None, (x+dx, y+dy),   # type: ignore
                          self.color, 4)  # type: ignore
        self.modified = True

    def do_autosave(self):
        if self.auto_save and self.modified:
            self.save()
            return True
        else:
            return False

    def save(self):
        filename = self.layer_filename()
        assert self.layer is not None
        if cv2.imwrite(filename, self.layer):
            print('Saved %s' % filename)
        self.modified = False

    def next(self):
        self.do_autosave()
        if self.index+1 < len(self.filenames):
            self.load(self.index+1)
        else:
            print('All images have been tagged.')

    def prev(self):
        self.do_autosave()
        if self.index-1 >= 0:
            self.load(self.index-1)
        else:
            print('First image.')


DEBUG_ARGS = ['tagger.py', '/data/river/tags/tags.2gelise-1-000060.png']

if __name__ == '__main__':
    # args = DEBUG_ARGS
    args = sys.argv
    print('Tagging %d images.' % len(args[1:]))
    editor = Editor(args[1:])
    editor.loop()
