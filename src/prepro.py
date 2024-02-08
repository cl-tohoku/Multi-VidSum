import argparse
from pathlib import Path

import preprocessors

def main(args, cls):
    preprocessor = cls(args)
    preprocessor()
    

if __name__ == "__main__":
    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument(
        "--preprocessor_name",
        help="Specify preprocessor_name",
        type=str,
        required=True
    )
    initial_args, unrecognized_arguments_str = initial_parser.parse_known_args()
    cls = getattr(preprocessors, initial_args.preprocessor_name)
    
    parser = argparse.ArgumentParser()
    cls.add_args(parser)
    args = parser.parse_args(unrecognized_arguments_str)
    main(args, cls)
