import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional
from collections import defaultdict
import subprocess
import json
import csv
from itertools import islice

from tqdm.auto import tqdm


def count_line(file_path):
    return int(
        subprocess.check_output(
            ["wc", "-l", file_path]
        ).decode().split(" ")[0]
    )


def main(args):
    dict_data = {}
    file_length = count_line(args.caption_csv_file_path)
    if args.without_header:
        file_length -= 1
    
    with args.caption_csv_file_path.open(mode="r") as f:
        if not args.without_header:
            header = next(f)
        
        for video_id, frame_index, image_path, caption in tqdm(
                csv.reader(f, delimiter=args.delimiter),
                total=file_length,
        ):
            if not args.without_v_prefix:
                video_id = video_id[2:]
            
            feature_file_path = args.feature_dir_path / f"{video_id}_{args.feature_model_name}.npy"
            if not feature_file_path.exists():
                continue
            
                
            if video_id not in dict_data:
                dict_data[video_id] = {
                    "frames": []
                }
            
            assert Path(image_path).exists(), f"{image_path} dose not exist.."
            dict_data[video_id]["frames"].append(
                {
                    "idx": int(frame_index),
                    "path": image_path,
                    "caption": caption,
                }
            )
    
    # sort and verification
    for video_id, data in tqdm(dict_data.items(), total=len(dict_data)):
        data["frames"].sort(key=lambda d: d["idx"])
        assert list(range(len(data["frames"]))) == [d["idx"] for d in data["frames"]]
    
    with args.output_json_file_path.open(mode="w") as f:
        json.dump(dict_data, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_csv_file_path", type=Path, required=True)
    parser.add_argument("--output_json_file_path", type=Path, required=True)
    parser.add_argument("--feature_dir_path", type=Path, required=True)
    parser.add_argument("--feature_model_name", type=str, default="clip")
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--without_header", action='store_true')
    parser.add_argument("--without_v_prefix", action='store_true')
    args = parser.parse_args()
    main(args)
