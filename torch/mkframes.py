#!/usr/bin/env python
# Extracts random frames from a set of videos, and resizes them.

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from cv2.typing import MatLike


def parse_size(arg: str) -> Tuple[int, int]:
    if (m := re.match(r'^\s*(\d+)x(\d+)\s*$', arg)):
        return int(m.group(1)), int(m.group(2))
    else:
        raise argparse.ArgumentTypeError(f"Invalid size: {arg},expecting WxH")
    return (width, height)
               

parser = argparse.ArgumentParser(
    prog='mkframes.py',
    description="Extract frames as images from a set of videos."
)               

parser.add_argument('-n', '--dry-run', action='store_true', default=False,
                    help='Frequency of extracted frames.')
parser.add_argument('-q', '--quiet', action='store_true', default=False,
                    help="Don't report any progress.")
parser.add_argument('-w', '--overwrite', action='store_true', default=False,
                    help="Don't overwrite existing images.")
parser.add_argument('-f', '--freq', type=int, default=30,
                    help='Frequency of extracted frames.')
parser.add_argument('-s', '--size', type=lambda s : parse_size(s), default=(640, 360),
                    metavar='WxH',
                    help="""Size of the extracted frames as WxH.""")
parser.add_argument('-d', '--dest', type=Path, default=Path(),
                    metavar='DEST',
                    help="Destination directory, created if it doesn't exist")
parser.add_argument('-t', '--test-directory', type=Path, default=None,
                    metavar='TEST_DIRECTORY',
                    help="Directory to use to test for file existence. "+
                        "Used in conjunction with -d DIR, it will check "+
                        "if the target images exists in TEST_DIRECTORY, and if  not will create it in DEST.")
parser.add_argument("files", type=Path, nargs="*", 
                    help="Videos to process.")
               
args = parser.parse_args()

class MovieReader(object):

    filenames: List[ str ]
    file_index: int
    frame_index: int
    
    def __init__(self, filenames: List[ str ]):
        self.filenames = filenames
        self.file_index = 0
        self.frame_index = 0
        self.video = None
        self.frame = None

    def reset(self):
        self.file_index = 0
        self.frame_index = 0
        self.video = None
        self.frame = None
        
    def get_frame_count(self):
        frame_count = 0
        video = None
        for filename in self.filenames:
            try:
                video = cv2.VideoCapture(filename)
                if video.isOpened():
                    frame_count += video.get(cv2.CAP_PROP_FRAME_COUNT)
            finally:
                if video is not None:
                    video.release()
        return frame_count

    def get_filename(self):
        return self.filenames[self.file_index]

    def get_frame_index(self):
        return self.frame_index

    def skip(self, count: int):
        while count > 0:
            count -= 1
            if self.next() is None:
                return None

    def get_frame(self):
        return self.frame
  
    def next(self) -> Optional[ MatLike ]:
        while True:
            # Opens next file if needed, returns when we've exhausted all files.
            while self.video is None and self.file_index < len(self.filenames):
                self.video = cv2.VideoCapture(self.filenames[self.file_index])
                self.frame_index = 0
                if not self.video.isOpened():
                    self.video = None
                    self.file_index += 1
            if self.video is None:
                return None
            # Grabs the next frame and returns it.
            ret, frame = self.video.read()
            self.frame_index += 1
            if ret and frame is not None:
                self.frame = frame
                return frame
            else:
                # Assume we've reached the end of this video, skip to next.
                self.video = None
                self.file_index += 1


def main():
    # Checks the output directory.
    if not args.dest.exists():
        args.dest.mkdir()
        
    reader = MovieReader(args.files)
    while True:
        frame = reader.next()
        if frame is None:
            break
        # Skips according to frequency.
        if (1+reader.get_frame_index()) % args.freq != 0:
            continue
            
        # Extracts the frames and settle it.
        frame = cv2.resize(frame, args.size)
        basename, _ = os.path.splitext(os.path.basename(reader.get_filename()))
        filename = f"{basename}-{1+reader.get_frame_index():06}.jpg"
        outpath = Path(args.dest, filename).as_posix()
        tstpath = Path(
            args.test_directory if args.test_directory is not None else args.dest,
            filename
        ).as_posix()
        # Says what we do, or do it.
        if not args.overwrite and os.path.exists(tstpath):
            if not args.quiet:
                print(f"! {outpath}")
        else:        
            if args.dry_run:
                print(f"+ {outpath}")
            elif not cv2.imwrite(outpath, frame):
                print(f"Failed to create file {outpath}")
            elif not args.quiet:
                    print(f"+ {outpath}")
            

if __name__ == '__main__':
    main()
