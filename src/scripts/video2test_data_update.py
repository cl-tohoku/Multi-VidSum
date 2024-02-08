import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional
from itertools import count, islice

import cv2
from PIL import Image
from tqdm.auto import tqdm

def get_frames(video_path):
    image_list = []
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    for frame_sec in map(lambda i: i * 0.5, count()):
        frame_idx = round(fps * frame_sec)

        if frame_count <= frame_idx:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        yield frame



def main(args):
    assert args.video_dir.exists()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_clip.npy"
    candidate_video_ids = set(
        [
            path.name[:-len(suffix)]
            for path in args.feature_dir.glob(f"*{suffix}")
        ]
    )

    pbar = tqdm(
        total=args.num_convert,
    )
    count = 0
    for path in args.video_dir.glob("*.mp4"):
        video_id = path.name[2:-4]
        if video_id not in candidate_video_ids:
            continue
        save_dir_path = args.output_dir / f"v_{video_id}"
        save_dir_path.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(get_frames(path)):
            cv2.imwrite(str(save_dir_path / f"{i}.jpg"), frame)
        
        pbar.update()
        count += 1
        if count == args.num_convert:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", help="Specify file path", type=Path)
    parser.add_argument("output_dir", help="Specify file path", type=Path)
    parser.add_argument("feature_dir", help="Specify file path", type=Path)
    parser.add_argument("--num_convert", "-n", type=int)
    args = parser.parse_args()
    main(args)
