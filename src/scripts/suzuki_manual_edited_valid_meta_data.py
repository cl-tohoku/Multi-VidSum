import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional
from itertools import groupby
import json

from tqdm.auto import tqdm
from logzero import logger

def main(args):
    meta_data = {}
    with args.original_meta_data_path.open(mode="r") as f:
        for json_dict in tqdm(map(json.loads, f)):
            video_id = json_dict.pop("video_name")[2:]
            if video_id not in meta_data:
                json_dict["split"] = [json_dict["split"]]
                meta_data[video_id] = json_dict
            else:
                meta_data[video_id]["split"].append(json_dict["split"])
                meta_data[video_id]["caption_info"].update(
                    json_dict["caption_info"]
                )

    count = 0
    with args.input_file_path.open(mode="r") as f_input, \
         args.output_file_path.open(mode="w") as f_output:
        for is_sep, data in groupby(f_input, key=lambda s: s.startswith("###")):
            if is_sep:
                continue
            
            output_josn = {
                "caption_info": {}
            }
            for line in data:
                video_id, sent_id, caption = line.split("\t")
                # rmove .html suffix
                video_id = video_id[:-5]
                feature_file_path = args.feature_dir_path / f"{video_id}_{args.feature_model_name}.npy"
                if not feature_file_path.exists():
                    logger.info(f"Skip (no video feature): {video_id}")
                    break
                
                if video_id not in meta_data:
                    logger.info(f"Skip (no data in meta data): {video_id}")
                    break
                
                output_josn["caption_info"][sent_id] = {
                    "caption": caption,
                    "segment_start": meta_data[video_id]["caption_info"][sent_id]["segment_start"],
                    "segment_end": meta_data[video_id]["caption_info"][sent_id]["segment_end"],
                    "anotated_times": meta_data[video_id]["caption_info"][sent_id]["anotated_times"]
                }
            else:
                output_josn["video_name"] = f"v_{video_id}"
                output_josn["split"] = meta_data[video_id]["split"]
                output_josn["movie_length"] = meta_data[video_id]["movie_length"]
                f_output.write(json.dumps(output_josn) + "\n")
                count += 1
    
    logger.info(f"Count: {count}")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=Path, required=True)
    parser.add_argument("--output_file_path", type=Path, required=True)
    parser.add_argument("--original_meta_data_path", type=Path, required=True)
    parser.add_argument("--feature_dir_path", type=Path, required=True)
    parser.add_argument("--feature_model_name", type=str, default="clip")

    # parser.add_argument("--use_gpu", action='store_true')
    args = parser.parse_args()
    main(args)
