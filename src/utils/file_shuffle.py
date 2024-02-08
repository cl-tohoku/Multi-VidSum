import random
from pathlib import Path

from more_itertools import ilen, chunked

def large_file_shuffle(input_file_path:Path, output_file_path:Path, batch_size=100000, seed=42):
    random.seed(seed)
    with input_file_path.open(mode="r") as f:
        line_count = ilen(f)
    
    indexes = list(range(line_count))
    random.shuffle(indexes)

    with output_file_path.open(mode="w") as f_output:

        for index_batch in chunked(indexes, batch_size):
            batch = {}
            index_batch_set = set(index_batch)
            with input_file_path.open(mode="r") as f_input:
                for i, line in enumerate(f_input):
                    if i in index_batch_set:
                        batch[i] = line

                for i in index_batch:
                    f_output.write(batch[i])

