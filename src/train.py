import argparse
from trainer import Trainer


def main(args):
    trainer = Trainer(args, mode=args.mode)
    trainer(train_only=args.train_only)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_args(parser)
    parser.add_argument(
        "--mode",
        help="Select mode",
        type=str,
        required=True,
        choices=["train", "resume"]
    )
    parser.add_argument(
        "--train_only",
        help="Specify whether to exec test at the same time",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)
