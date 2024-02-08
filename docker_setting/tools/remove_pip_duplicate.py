import argparse
from pathlib import Path
import re

def yes_no_input(path):
    while True:
        choice = input(f"Output path \"{path}\"is already exist. Over write? [y/n]: ").lower()
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False


def extract_lib_name(line):
    signs = ["==", ">=", "<=", "!=", "~=", "@"]

    for sign in signs:
        re_match_obj = re.search(sign, line)
        if re_match_obj is not None:
            start, _ = re_match_obj.span()
            return line[:start].strip()

    return line.strip()

            
        
def main(args):
    if args.output_file_path.exists() and (not yes_no_input(args.output_file_path)):
        return


    with args.already_installed_pip_file_path.open(mode="r") as f:
        installled_lib_name = [extract_lib_name(line.strip()) for line in f]
        
    with args.required_pip_file_path.open(mode="r") as f_required,\
         args.output_file_path.open(mode="w") as f_output:

        for line in f_required:
            lib_name = extract_lib_name(line.rstrip())
            if lib_name in installled_lib_name:
                continue

            f_output.write(line)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--required_pip_file_path", type=Path, required=True)
    parser.add_argument("--already_installed_pip_file_path", type=Path, required=True)
    parser.add_argument("--output_file_path", type=Path, default="requirements.txt")
    args = parser.parse_args()
    main(args)
