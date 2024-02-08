import argparse
from pathlib import Path

from trainer import Trainer


def main(args):
    trainer = Trainer(args, mode="test")
    trainer.test()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_args(parser)
    args = parser.parse_args()
    main(args)
