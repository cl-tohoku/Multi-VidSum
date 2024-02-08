from itertools import groupby, dropwhile
from pathlib import Path
import json

from logzero import logger
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import cv2
from PIL import Image

class Video2TextT5PL(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("pl_module_setting")

        # decode configurations
        parser.add_argument("--num_beams", type=int, default=1)
        parser.add_argument("--max_length", type=int, default=512)
        parser.add_argument("--length_penalty", type=float, default=0.0)

        # Model architecuture arguments
        parser.add_argument("--dropout_rate", type=float)
        parser.add_argument("--is_pointer_network", action="store_true")
        parser.add_argument("--with_gate", action="store_true")
        parser.add_argument("--with_discrete_gate", action="store_true")
        parser.add_argument("--normalize_last_hidden_states", action="store_true")
        parser.add_argument("--label_smoothing", type=float, default=0.0)
        parser.add_argument("--separated_cross_entropy_loss", action="store_true")
        parser.add_argument("--loss_frame_weight", type=float, default=0.5)
        
        # lr scheduler configurations
        parser.add_argument("--lr", type=float, required=True)
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            choices=["cosine_scheduler", "ReduceLROnPlateau"],
        )
        parser.add_argument(
            "--lr_scheduler_interval",
            type=str,
            default="step",
            choices=["step", "epoch"],
        )
        parser.add_argument("--num_warmup_steps", type=int)
        

        # Options for ReduceLROnPlateau
        parser.add_argument("--lr_reduce_mode", type=str, choices=["min", "max"])
        parser.add_argument("--lr_reduce_factor", type=float, default=0.1)
        parser.add_argument("--lr_reduce_patience", type=int, default=1)
        parser.add_argument("--lr_reduce_threshold", type=float, default=1e-4)
        parser.add_argument(
            "--lr_reduce_threshold_mode",
            type=str,
            default="rel",
            choices=["rl", "abs"]
        )
        parser.add_argument("--min_lr", type=float, default=0.0)
        parser.add_argument("--lr_reduce_eps", type=float, default=0.0)
        parser.add_argument("--lr_reduce_monitor", type=str, default="valid_loss")
        
        # AdamW configurations
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--eps", type=float, default=1e-08)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        
        # Model, Tokenizer initialization configurations
        parser.add_argument("--model_name_or_path", help="Select model name or path", type=str, required=True)
        parser.add_argument("--tokenizer_name_or_path", help="Select model name or path", type=str, required=True)
        parser.add_argument("--from_scratch", help="Select whether to use pretrained model", action="store_true")
        parser.add_argument(
            "--model_max_length",
            help="Specify number of max input tokens in encoder.",
            type=int,
        )


        # Additional configurations
        parser.add_argument(
            "--discretized_frequency",
            help="Specify video discretized frequency",
            type=float,
        )

        # Evaluation configurations
        parser.add_argument(
            "--num_wandb_upload",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--bleu_tokenizer",
            type=str,
            default="13a",
        )
        parser.add_argument(
            "--joint_evaluation",
            help="Specify whether to join text in evaluation",
            action="store_true",
        )
        parser.add_argument(
            "--video_dir",
            help="Specify directory that contains video.",
            type=Path,
        )
        return parser

    
    def __init__(self, config):
        super().__init__()
        self.test_data_idx = 0
        self.automatic_optimization = True
        self.save_hyperparameters(config)
        
        if self.hparams.from_scratch:
            self.model_config = T5Config.from_pretrained(self.hparams.model_name_or_path)
            self.overwrite_model_config()
            self.model = Video2TextConditionalGeneration(config=self.model_config)
            logger.info("Learning from scrach!")
        else:
            self.model_config = T5Config.from_pretrained(self.hparams.model_name_or_path)
            self.overwrite_model_config()
            self.model = Video2TextConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path,
                config=self.model_config
            )
            logger.info(f"Load pretrained model from \"{self.hparams.model_name_or_path}\"")
        
        self._post_init()

        
    def _post_init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name_or_path
        )
        self.sep_token_id =  self.tokenizer.pad_token_id if self.tokenizer.bos_token_id is None else self.tokenizer.bos_token_id
        assert self.tokenizer.vocab_size == self.model_config.vocab_size, \
            "Vocab size for tokenizer and model must be same!"
        
        if self.hparams.model_max_length is not None:
            self.tokenizer.model_max_length = self.hparams.model_max_length
        assert self.tokenizer.model_max_length == self.model.config.n_positions, \
            f"config.n_positions: {self.model.config.n_positions} != tokenizer.model_max_length: {self.tokenizer.model_max_length}"
        
    def overwrite_model_config(self):
        if self.hparams.dropout_rate is not None:
            self.model_config.dropout_rate = self.hparams.dropout_rate
            
        self.model_config.is_pointer_network = self.hparams.is_pointer_network
        assert (not self.hparams.with_gate) or self.model_config.is_pointer_network
        assert not (self.hparams.with_discrete_gate and self.hparams.with_gate)
        self.model_config.with_gate = self.hparams.with_gate
        self.model_config.with_discrete_gate = self.hparams.with_discrete_gate
        self.model_config.normalize_last_hidden_states = self.hparams.normalize_last_hidden_states
        self.model_config.label_smoothing = self.hparams.label_smoothing
        self.model_config.loss_frame_weight = self.hparams.loss_frame_weight
        self.model_config.separated_cross_entropy_loss = self.hparams.separated_cross_entropy_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.lr_scheduler == "cosine_scheduler":
            assert self.hparams.num_warmup_steps is not None
            lr_sheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            lr_sheduler_config = {
                "scheduler": lr_sheduler,
                "interval": self.hparams.lr_scheduler_interval,
                "frequency": 1,
            }
        elif self.hparams.lr_scheduler == "ReduceLROnPlateau":
            lr_sheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode=self.hparams.lr_reduce_mode,
                factor=self.hparams.lr_reduce_factor,
                patience=self.hparams.lr_reduce_patience,
                threshold=self.hparams.lr_reduce_threshold,
                threshold_mode=self.hparams.lr_reduce_threshold_mode,
                min_lr=self.hparams.min_lr,
                eps=self.hparams.lr_reduce_eps,
            )
            lr_sheduler_config = {
                "scheduler": lr_sheduler,
                "interval": self.hparams.lr_scheduler_interval,
                "frequency": 1,
                "monitor": self.hparams.lr_reduce_monitor,
                "strict": True,
            }
        else:
            raise ValueError()


        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_sheduler_config,
        }

    
    def training_step(self, batch, batch_idx=None):
        batch.pop("video_id")
        batch.pop("frame_idxs")
        output = self.model(**batch, output_hidden_states=False)
        self.log("train_loss", output.loss.item(), on_step=True, on_epoch=True)
        return {"loss": output.loss}


    def validation_step(self, batch, batch_idx=None):
        batch.pop("video_id")
        batch.pop("frame_idxs")
        
        output = self.model(**batch, output_hidden_states=False)
        self.log(
            "valid_loss",
            output.loss.item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
    
    def test_step(self, batch, batch_idx=None):
        video_ids = batch.pop("video_id")
        frame_idxs = batch.pop("frame_idxs")
        output = self.model(**batch, output_hidden_states=False)
        
        torch.use_deterministic_algorithms(False)
        decoded_ids = self.model.generate(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            num_beams=self.hparams.num_beams,
            max_length=self.hparams.max_length,
            do_sample=False,
            min_length=0,
            num_beam_groups=1,
            no_repeat_ngram_size=0,
            encoder_no_repeat_ngram_size=0,
            length_penalty=self.hparams.length_penalty,
        )
        torch.use_deterministic_algorithms(True)
        
        # model.generate include decoder start token id in generated tokens
        hypothesis = self.decode_and_frame_select(decoded_ids[:, 1:].tolist())
        references = self.decode_and_frame_select(batch["labels"].tolist())
        self.log(
            f"test_loss_{self.test_data_idx}",
            output.loss.item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        
        assert len(video_ids) == len(hypothesis) == len(references)
        return {
            "video_ids": video_ids,
            "hypothesis": hypothesis,
            "references": references,
        }


    def test_epoch_end(self, outputs):
        normalized_hyp_refs = []
        for output_batch in outputs:
            key_list = list(output_batch.keys())
            for values in zip(*tuple(output_batch.values())):
                assert len(key_list) == len(values)
                output_instance = {k: v for k, v in zip(key_list, values)}
                
                # Pad texts
                ref_texts = output_instance["references"]["texts"]
                hyp_texts = output_instance["hypothesis"]["texts"]
                if len(ref_texts) < len(hyp_texts):
                    ref_texts = ref_texts + [
                        "" for i in range(len(hyp_texts) - len(ref_texts))
                    ]
                elif len(ref_texts) > len(hyp_texts):
                    hyp_texts = hyp_texts + [
                        "" for i in range(len(ref_texts) - len(hyp_texts))
                    ]
                assert len(ref_texts) == len(hyp_texts)
                
                # Pad frame idxs
                ref_frame_idxs = output_instance["references"]["frame_idxs"]
                hyp_frame_idxs = output_instance["hypothesis"]["frame_idxs"]
                if len(ref_frame_idxs) < len(hyp_frame_idxs):
                    ref_frame_idxs = ref_frame_idxs + [
                        self.tokenizer.vocab_size + self.model_config.n_positions
                        for _ in range(len(hyp_frame_idxs) - len(ref_frame_idxs))
                    ]
                elif len(ref_frame_idxs) > len(hyp_frame_idxs):
                    hyp_frame_idxs = hyp_frame_idxs + [
                        self.tokenizer.vocab_size + self.model_config.n_positions
                        for _ in range(len(ref_frame_idxs) - len(hyp_frame_idxs))
                    ]
                assert len(ref_frame_idxs) == len(hyp_frame_idxs)

                normalized_hyp_refs.append(
                    {
                        "video_id": output_instance["video_ids"],
                        "ref_texts": ref_texts,
                        "hyp_texts": hyp_texts,
                        # Huck to prevent converting to tensor
                        "ref_frame_idxs": list(map(str, ref_frame_idxs)), 
                        "hyp_frame_idxs": list(map(str, hyp_frame_idxs)),
                    }
                )
        
        normalized_hyp_refs = self.all_gather(normalized_hyp_refs)
        if self.trainer.is_global_zero:
            # Calc BLEU score
            if self.hparams.joint_evaluation:
                hyp_texts = [
                    " ".join(instance["hyp_texts"])
                    for instance in normalized_hyp_refs
                ]
                ref_texts = [
                    " ".join(instance["ref_texts"])
                    for instance in normalized_hyp_refs
                ]
            else:
                hyp_texts = sum(
                    (instance["hyp_texts"] for instance in normalized_hyp_refs),
                    start=[]
                )
                ref_texts = sum(
                    (instance["ref_texts"] for instance in normalized_hyp_refs),
                    start=[]
                )
            
            bleu = BLEU(
                tokenize=self.hparams.bleu_tokenizer,
            )
            
            bleu_score = bleu.corpus_score(
                hyp_texts,
                [
                    ref_texts,
                ]
            )
            sacrebleu_result = json.loads(
                bleu_score.format(
                    signature=str(bleu.get_signature()),
                    width=10,
                    is_json=True
                )
            )

            # Calc frame accuracy
            hyp_frame_idxs = sum(
                (list(map(int, instance["hyp_frame_idxs"])) for instance in normalized_hyp_refs),
                start=[]
            )
            ref_frame_idxs = sum(
                (list(map(int, instance["ref_frame_idxs"])) for instance in normalized_hyp_refs),
                start=[]
            )

            frame_accuracy = accuracy_score(
                hyp_frame_idxs,
                ref_frame_idxs,
            )
            
            
            # Logging
            sacrebleu_result_path = self.hparams.log_dir / "sacrebleu.json"
            with sacrebleu_result_path.open(mode="w") as f:
                json.dump(sacrebleu_result, f, ensure_ascii=False, indent=4)
            self.logger.experiment.save(str(sacrebleu_result_path))
            
            logging_metrics = {}
            for k, v in sacrebleu_result.items():
                try:
                    num = float(v)
                except ValueError:
                    continue
                logging_metrics[f"test_{k}_{self.test_data_idx}"] = num
            logging_metrics[f"test_frame_accuracy_{self.test_data_idx}"] = frame_accuracy

            table_data = []
            for data in normalized_hyp_refs[:self.hparams.num_wandb_upload]:
                combined_frame_secs = [
                    idx * self.hparams.discretized_frequency
                    for idx in map(int, data["ref_frame_idxs"] + data["hyp_frame_idxs"])
                ]
                frames = self.get_frames(
                    self.hparams.video_dir / f"v_{data['video_id']}.mp4",
                    combined_frame_secs,
                )
                
                frames = [
                    Image.new(mode="RGB", size=(1,1)) if im is None else self.cv2pil(im)
                    for im in frames
                ]
                wandb_images = list(map(wandb.Image, frames))
                map(lambda im: im.close(), frames)
                
                table_data.append(
                    [
                        data["video_id"],
                        wandb_images[:len(data["ref_frame_idxs"])],
                        wandb_images[-len(data["hyp_frame_idxs"]):],
                        data["ref_texts"],
                        data["hyp_texts"],
                        list(map(int, data["ref_frame_idxs"])),
                        list(map(int, data["hyp_frame_idxs"])),
                    ]
                )
            
            self.logger.log_table(
                key="Predictions",
                columns=[
                    "video_id",
                    "ref_frame","hyp_frame",
                    "ref_text", "hyp_text",
                    "ref_frame_idx", "hyp_frame_idx", 
                ],
                data=table_data,
            )
            
            self.log_dict(
                logging_metrics,
                sync_dist=True,
                rank_zero_only=True
            )
            
    def decode_and_frame_select(self, decoded_ids):
        outputs = []
        for ids in map(self.remove_pad, decoded_ids):

            if len(ids) == 0:
                outputs.append(
                    {
                        "texts": [],
                        "frame_idxs": [],
                    }
                )
                continue
            
            # 最初の出力がframeじゃない時, ダミーのidを挿入する
            if ids[0] < self.tokenizer.vocab_size:
                ids.insert(0, self.tokenizer.vocab_size + self.model_config.n_positions)

            frames = []
            texts = []
            for is_text_part, id_iter in groupby(
                    ids, key=lambda j: j < self.tokenizer.vocab_size
            ):
                id_list = list(id_iter)
                if is_text_part:
                    text = self.tokenizer.decode(
                        id_list,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    texts.append(text)
                else:
                    # フレームが2つ連続している時は，最初のフレームのみが評価に使われる
                    frame_index = id_list[0] - self.tokenizer.vocab_size
                    frames.append(frame_index)

            # 最後の出力がフレームの時は無視
            if len(frames) == len(texts) + 1:
                frames = frames[:-1]
            assert len(texts) == len(frames)
            
            outputs.append(
                {
                    "texts": texts,
                    "frame_idxs": frames
                }
            )
        
        return outputs

    # <急募> もっと綺麗な実装
    def remove_pad(self, ids):
        return list(
            reversed(
                list(
                    dropwhile(
                        lambda i: (i == self.tokenizer.pad_token_id) or (i == -100),
                        list(reversed(ids)),
                    )
                )
            )
        )
    
    def get_frames(self, video_path, frame_sec_list):
        image_list = []
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened()
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for frame_sec in frame_sec_list:
            frame_idx = round(fps * frame_sec)
            
            if frame_count <= frame_idx:
                image_list.append(None)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            image_list.append(frame)
            
        return image_list
    
    # Copied from https://qiita.com/derodero24/items/f22c22b22451609908ee
    def cv2pil(self, image):
        ''' OpenCV型 -> PIL型 '''
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = Image.fromarray(new_image)
        return new_image
