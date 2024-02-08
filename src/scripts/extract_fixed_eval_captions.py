import argparse
from pathlib import Path
import json

def main(args):
    with args.input_file_path.open() as f:
        data = json.load(f)

    video_sets = set()
    with args.video_list_file_path.open() as f:
        video_sets = {line.rstrip() for line in f}

    extracted_data = {}
    for k, v in data.items():
        if k in video_sets:
            extracted_data[k] = v

    # Validate extracted_data
    for vid in video_sets:
        assert vid in extracted_data, f"{vid} is not in extracted_data"

    args.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file_path.open("w") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", help="Specify file path", type=Path)
    parser.add_argument("--output_file_path", help="Specify file path", type=Path)
    parser.add_argument("--video_list_file_path", help="Specify file path", type=Path)
    #parser.add_argument("--use_gpu", action='store_true')
    args = parser.parse_args()
    main(args)
