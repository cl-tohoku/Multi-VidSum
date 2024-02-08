import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import models

def main(args):
    cls = getattr(models, args.pl_model_name)
    pl_model = cls.load_from_checkpoint(
         args.ckpt_file_path,
    )
    args.weigh_save_dir.mkdir(exist_ok=True, parents=True)
    pl_model.model.save_pretrained(args.weigh_save_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pl_model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ckpt_file_path",
        help="Specify file path",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--weigh_save_dir",
        help="Specify file path",
        type=Path
    )
    #parser.add_argument("--use_gpu", action='store_true')
    args = parser.parse_args()
    main(args)
