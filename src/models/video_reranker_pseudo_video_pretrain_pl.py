from itertools import groupby, dropwhile, islice
from pathlib import Path
import json

from logzero import logger
import torch
from torch.nn.functional import one_hot
import pytorch_lightning as pl
from transformers import (
    get_cosine_schedule_with_warmup,
    AutoTokenizer,
    T5Config,
    Video2TextConditionalGeneration,
)
import wandb
from sacrebleu.metrics import BLEU
from sklearn.metrics import accuracy_score
from PIL import Image

from .video_reranker_t5_pl import Video2TextRerankerT5PL

class Video2TextRerankerPseudoVideoPretrainT5PL(Video2TextRerankerT5PL):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Video2TextRerankerT5PL.add_model_specific_args(parent_parser)
        parser.add_argument("--apply_noise", action="store_true")
        parser.add_argument("--noise_intensity", type=float, default=0.05)
        return parser
    
    def forward(self, batch):
        if self.hparams.apply_noise:
            frame_idx_one_hot = one_hot(
                batch["frame_idxs"].view(-1),
                num_classes=batch["inputs_embeds"].size(1) + 1,
            )
                    
            # Remove pad
            frame_idx_one_hot = frame_idx_one_hot[:, :-1]
            noise_mask = 1 - frame_idx_one_hot.view(
                -1,
                batch["frame_idxs"].size(-1),
                frame_idx_one_hot.size(-1),
            ).sum(-2)
            noise = torch.randn_like(batch["inputs_embeds"]) * torch.mean(batch["inputs_embeds"]) * self.hparams.noise_intensity
            noise = torch.einsum("bsd, bs->bsd", noise, noise_mask)
            batch["inputs_embeds"] = batch["inputs_embeds"] + noise
            
        return super().forward(batch)
    
