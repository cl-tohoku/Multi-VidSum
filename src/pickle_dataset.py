from pathlib import Path
import pickle
import json

from logzero import logger
from torch.utils.data import IterableDataset, Dataset

import collators
from utils.pickle_file import PickleFileLoader
    

class PickleDataset(Dataset):
    def __init__(self, dataset_path:Path):
        self.dataset_path = dataset_path
        self.basic_info_path = self.dataset_path / "basic_info.json"
        self.tokenized_data_file_path = self.dataset_path / "tokenized_data.pkl"
        
        with self.basic_info_path.open(mode="r") as f:
            self.basic_info = json.load(f)
        
        self.collator = getattr(collators, self.basic_info["collator"])

        self.tokenized_data = [
            obj for obj in PickleFileLoader(self.tokenized_data_file_path)
        ] 
        
    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, item):
        return self.tokenized_data[item]
    
    
