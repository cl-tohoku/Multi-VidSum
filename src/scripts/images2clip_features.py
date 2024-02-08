import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional
import multiprocessing
from itertools import islice
import time

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset
import clip
from logzero import logger
from tqdm.auto import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImagePathDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.path_list = [
            Path(s.rstrip())
            for s in args.path_list_file_path.open(mode="r")
        ]
        file_names = [path.name for path in self.path_list]
        assert len(file_names) == len(set(file_names)), "There are some file name duplicates"
        self.path_list = self.path_list[args.job_id::args.world_size]
        
    def __getitem__(self, idx):
        return self.path_list[idx]

    def __len__(self):
        return len(self.path_list)



class CLIPCollator:
    def __init__(self, args, preprocess):
        self.args = args
        self.preprocess = preprocess

    def __call__(self, batch_source):
        images = [
            Image.open(path)
            for path in batch_source
        ]
        return {
            "inputs": torch.stack([self.preprocess(im) for im in images]),
            "save_paths": [
                args.output_dir / f"{path.stem}_clip.npy"
                for path in batch_source
            ],
        }



def save_feature(worker_index, queue):
    while True:
        data = queue.get()
        for feature, path in zip(
                data["fetuares"],
                data["save_paths"],
        ):
            np.save(path, feature)
        queue.task_done()
        
def create_worker(worker_index, queue):
    return multiprocessing.Process(
        target=save_feature,
        args=(worker_index, queue)
    )
    

@torch.no_grad()
def main(args):
    assert args.job_id < args.world_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}")
    dataset = ImagePathDataset(args)

    queue = multiprocessing.JoinableQueue()
    workers = []
    for i in range(args.num_output_workers):
        write_worker = create_worker(i, queue)
        write_worker.start()
        workers.append(write_worker)
    
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_input_workers,
        collate_fn=CLIPCollator(args, preprocess),
        pin_memory=True,
    )
    num_loop = (len(dataset) // args.batch_size) + bool(len(dataset) % args.batch_size)
    for batch in tqdm(data_loader, total=num_loop):
        batch["inputs"] = batch["inputs"].to(device)
        features = model.encode_image(batch["inputs"])
        queue.put(
            {
                "fetuares": features.cpu().numpy(),
                "save_paths": batch["save_paths"],
            }
        )
    
    logger.info("Finish feature generations. Wait saving..!")
    queue.join()
    for worker in workers:
        worker.terminate()
    logger.info("Finish!")
             

             
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_list_file_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--num_input_workers", type=int, default=0)
    parser.add_argument("--num_output_workers", type=int, default=1)
    args = parser.parse_args()
    main(args)

