from pathlib import Path
import json
from datetime import datetime

from logzero import logger
import torch
from torch.nn.functional import one_hot, gumbel_softmax, cosine_similarity, cross_entropy
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

from .video_t5_pl import Video2TextT5PL

class Video2TextRerankerT5PL(Video2TextT5PL):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Video2TextT5PL.add_model_specific_args(parent_parser)

        parser.add_argument("--fix_embedding", action="store_true")
        parser.add_argument("--iterative_beam_search", action="store_true")
        parser.add_argument(
            "--num_key_frame",
            help="Specify minimum number of key frame. if not specified, all anotated frame will be used.",
            type=int,
        )
        
        parser.add_argument("--use_frame_loss", action="store_true")
        parser.add_argument("--frame_loss_alpha", type=float, default=1.0)
        parser.add_argument("--gumbel_softmax_tau", type=float, default=0.1)
        parser.add_argument("--multi_gold_probability_distribution_labels", action="store_true")
        # parser.add_argument("--label_smoothing", type=float, default=0.0)
        return parser
        
    def _post_init(self):
        super()._post_init()
        # Freeze embedding (or not)
        embedding = self.model.get_input_embeddings()
        embedding.weight.requires_grad = not self.hparams.fix_embedding
        assert (not self.hparams.iterative_beam_search) or (self.hparams.batch_size == 1)
        
    @torch.no_grad()
    def get_frame_idx_and_location_one_hot(self, batch):
        frame_idx_one_hot = one_hot(
            batch["frame_idxs"],
            num_classes=batch["inputs_embeds"].size(1) + 1,
        )
        # Remove pad
        frame_idx_one_hot = frame_idx_one_hot[:, :, :-1]
        frame_loc_one_hot = one_hot(
            batch["frame_locations"],
            num_classes=batch["labels"].size(-1) + 1
        )
        # Remove pad
        frame_loc_one_hot = frame_loc_one_hot[:, :, :-1]
        return frame_idx_one_hot, frame_loc_one_hot
    
    
    def overwrite_decoder_input_embed_with_frame_feature(
            self,
            batch,
            frame_idx_one_hot,
            frame_loc_one_hot,
    ):
        with torch.no_grad():
            overwrite_features = torch.bmm(
                batch["inputs_embeds"].permute(0, 2, 1) if self.model.encoder_embedding_projection is None else self.model.encoder_embedding_projection(batch["inputs_embeds"]).permute(0, 2, 1),
                frame_idx_one_hot.to(batch["inputs_embeds"].dtype).permute(0, 2, 1),
            ).permute(0, 2, 1)

            erase_matrix = 1 - torch.sum(
                frame_loc_one_hot,
                dim=-2
            )


        embedding = self.model.get_input_embeddings()
        decoder_input_text_embed = embedding(
            torch.masked_fill(
                batch["labels"],
                batch["labels"] == -100,
                self.tokenizer.pad_token_id,
            )
        )

        with torch.no_grad():
            erased_decoder_input_embed = torch.einsum(
                "bsd,bs->bsd",
                decoder_input_text_embed,
                erase_matrix
            )
            overwrite_feature_hot = torch.einsum(
                "bfd,bfs->bsd",
                overwrite_features,
                frame_loc_one_hot.to(overwrite_features.dtype),
            )
            decoder_input_embed = erased_decoder_input_embed + overwrite_feature_hot
        return decoder_input_embed
    
    @torch.no_grad()
    def shift_embed_right(self, embed):
        embedding = self.model.get_input_embeddings()
        decoder_start_token_embeds = embedding(
            torch.tensor(
                [self.model.config.decoder_start_token_id],
                dtype=torch.long,
                device=self.device,
            )
        ).repeat_interleave(embed.size(0), dim=0).unsqueeze(1)
        return torch.cat((decoder_start_token_embeds, embed[:, :-1, :]), dim=1)
    
    
    @torch.no_grad()
    def overwrite_labels_with_frame_token_id(
            self,
            batch,
            frame_loc_one_hot,
    ):
        frame_loc_hot = torch.sum(
            frame_loc_one_hot,
            dim=1,
        )
        text_token_ids = torch.mul(batch["labels"], (1 - frame_loc_hot))
        frame_token_ids = torch.sum(
            (
                frame_loc_one_hot.view(-1, frame_loc_one_hot.size(-1)).T * (
                    batch["frame_idxs"].view(-1) + self.tokenizer.vocab_size
                )
            ).T.view(
                -1,
                batch["frame_idxs"].size(-1),
                batch["labels"].size(-1),
            ),
            dim=1,
        )
        return text_token_ids + frame_token_ids


    @torch.no_grad()
    def generate_multiple_gold_frame_mask(
            self,
            batch,
            frame_loc_one_hot
    ):
        breakpoint()
        anotated_time_one_hot = one_hot(
            batch["anotated_time_idxs"],
            num_classes=batch["inputs_embeds"].size(1) + 1,
        )
        # remove pad index
        anotated_time_hot = anotated_time_one_hot[:, :, :, :-1].sum(dim=-2)

        # from IPython import embed; embed()
        
        frame_part_mask = torch.einsum(
            "bln,bls->bsn",
            anotated_time_hot.to(torch.float),
            frame_loc_one_hot.to(torch.float)
        )
        frame_part_mask = (frame_part_mask + 0.5).long().bool()
        
        multiple_gold_mask = torch.zeros(
            frame_part_mask.size(0),
            frame_part_mask.size(1),
            self.tokenizer.vocab_size + self.model_config.n_positions,
            dtype=frame_part_mask.dtype,
            device=frame_part_mask.device,
        )
        multiple_gold_mask[
            :, :, self.tokenizer.vocab_size:self.tokenizer.vocab_size + frame_part_mask.size(-1)
        ] = frame_part_mask
        # bsz * label_length * (vocab_size + n_posistions)
        return multiple_gold_mask


    def make_multi_gold_probability_distribution_labels(
            self,
            batch,
            frame_loc_one_hot,
    ):

        # anotated_time_idxs: # bsz * max_number_of_captions * max_number_of_anotated_time
        
        
        frame_loc_hot = torch.sum(
            frame_loc_one_hot,
            dim=1,
        )
        text_token_ids = torch.mul(batch["labels"], (1 - frame_loc_hot))
        frame_token_ids = torch.sum(
            (
                frame_loc_one_hot.view(-1, frame_loc_one_hot.size(-1)).T * (
                    batch["frame_idxs"].view(-1) + self.tokenizer.vocab_size
                )
            ).T.view(
                -1,
                batch["frame_idxs"].size(-1),
                batch["labels"].size(-1),
            ),
            dim=1,
        )
        return text_token_ids + frame_token_ids        
    
    def frame_loss(
            self,
            frame_logits,
            batch,
            frame_idx_one_hot,
            frame_loc_one_hot
    ):
        with torch.no_grad():
            overwrite_features = torch.bmm(
                batch["inputs_embeds"].permute(0, 2, 1),
                frame_idx_one_hot.to(batch["inputs_embeds"].dtype).permute(0, 2, 1),
            ).permute(0, 2, 1)
            overwrite_feature_hot = torch.einsum(
                "bfd,bfs->bsd",
                overwrite_features,
                frame_loc_one_hot.to(overwrite_features.dtype),
            )
        
        frame_pseudo_one_hot = gumbel_softmax(
            frame_logits[:, :, :batch["inputs_embeds"].size(1)],
            tau=self.hparams.gumbel_softmax_tau,
            dim=-1,
        )
        selected_frame_features = torch.einsum(
            "bfd, bsf->bsd",
            batch["inputs_embeds"],
            frame_pseudo_one_hot,
        )
        cos_sim = cosine_similarity(
            overwrite_feature_hot,
            selected_frame_features,
            dim=-1,
        ).view(-1)
        frame_pos_mask = torch.sum(
            frame_loc_one_hot,
            dim=1,
        ).view(-1)
        return 1.0 - (torch.sum(cos_sim * frame_pos_mask) / torch.sum(frame_pos_mask))

    
    def forward(self, batch):
        frame_idx_one_hot, frame_loc_one_hot = self.get_frame_idx_and_location_one_hot(
            batch
        )
        batch["decoder_inputs_embeds"] = self.shift_embed_right(
            self.overwrite_decoder_input_embed_with_frame_feature(
                batch,
                frame_idx_one_hot=frame_idx_one_hot,
                frame_loc_one_hot=frame_loc_one_hot,
            )
        )
        
        if self.hparams.multi_gold_probability_distribution_labels:
            batch["labels"] = batch["label_distribution"]
            ignore_label_distribution_mask = (batch["labels"] == -100)
        else:
            batch["labels"] = self.overwrite_labels_with_frame_token_id(
                batch,
                frame_loc_one_hot=frame_loc_one_hot,
            )
            ignore_label_distribution_mask = None
            
        if "anotated_time_idxs" in batch:
            assert not self.hparams.multi_gold_probability_distribution_labels
            multiple_gold_mask = self.generate_multiple_gold_frame_mask(
                batch,
                frame_loc_one_hot,
            )
        else:
            multiple_gold_mask = None
            
        # debug_conf = {
        #     "inputs_embeds": batch["inputs_embeds"].size(),
        #     "attention_mask": batch["attention_mask"].size(),
        #     "decoder_inputs_embeds": batch["decoder_inputs_embeds"].size(),
        #     "labels": batch["labels"].size(),
        # }
        # print(
        #     f"Rank {self.global_rank} batch size: {json.dumps(debug_conf)}",
        #     flush=True,
        # )
        output = self.model(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            decoder_inputs_embeds=batch["decoder_inputs_embeds"],
            labels=batch["labels"],
            ignore_label_distribution_mask=ignore_label_distribution_mask,
            multiple_gold_mask=multiple_gold_mask,
            frame_location_hot = frame_loc_one_hot.sum(
                dim=1
            ) if self.hparams.with_discrete_gate or self.hparams.separated_cross_entropy_loss else None,
            output_hidden_states=False,
            output_attentions=False,
        )
        
        # Calculate frame similarity loss
        if self.hparams.use_frame_loss:
            output.loss += self.hparams.frame_loss_alpha * self.frame_loss(
                output.logits[:, :, self.model_config.vocab_size:],
                batch,
                frame_idx_one_hot=frame_idx_one_hot,
                frame_loc_one_hot=frame_loc_one_hot,
            )
        
        return output

        
    
    def training_step(self, batch, batch_idx=None):
        try:
            batch.pop("video_id")
        except KeyError:
            pass
        output = self(batch)
        self.log("train_loss", output["loss"].item(), on_step=True, on_epoch=True)
        return {"loss": output["loss"]}
    
    
    def validation_step(self, batch, batch_idx=None):
        try:
            batch.pop("video_id")
        except KeyError:
            pass
        output = self(batch)
        self.log(
            "valid_loss",
            output["loss"].item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )


    def iterative_beam_search(self, batch):
        torch.use_deterministic_algorithms(False)
        eos_token_ids = [self.sep_token_id for i in range(self.hparams.num_key_frame)]
        eos_token_ids[-1] = self.tokenizer.eos_token_id
        for i, current_eos_token_id in enumerate(eos_token_ids):
            if i == 0:
                decoder_input_ids = self.model.generate(
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
                    eos_token_id=current_eos_token_id,
                )
            else:
                decoder_input_ids = self.model.generate(
                    inputs_embeds=batch["inputs_embeds"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    num_beams=self.hparams.num_beams,
                    max_length=self.hparams.max_length,
                    do_sample=False,
                    min_length=0,
                    num_beam_groups=1,
                    no_repeat_ngram_size=0,
                    encoder_no_repeat_ngram_size=0,
                    length_penalty=self.hparams.length_penalty,
                    eos_token_id=current_eos_token_id,
                )
                
        torch.use_deterministic_algorithms(True)
        return decoder_input_ids
    
    def test_step(self, batch, batch_idx=None):
        video_ids = batch.pop("video_id")
        output = self(batch)
        
        if self.hparams.iterative_beam_search:
            decoded_ids = self.iterative_beam_search(batch)
        else:
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
            output["loss"].item(),
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
                    ref_frame_idxs += [
                        self.tokenizer.vocab_size + self.model_config.n_positions
                        for _ in range(len(hyp_frame_idxs) - len(ref_frame_idxs))
                    ]
                elif len(ref_frame_idxs) > len(hyp_frame_idxs):
                    hyp_frame_idxs += [
                        self.tokenizer.vocab_size + self.model_config.n_positions
                        for _ in range(len(ref_frame_idxs) - len(hyp_frame_idxs))
                    ]
                assert len(ref_frame_idxs) == len(hyp_frame_idxs)

                normalized_hyp_refs.append(
                    {
                        "video_id": output_instance["video_ids"],
                        "ref_texts": ref_texts,
                        "hyp_texts": hyp_texts,
                        # Huck to prevent being converted to tensor
                        "ref_frame_idxs": ref_frame_idxs, 
                        "hyp_frame_idxs": hyp_frame_idxs,
                    }
                )


        
        temp_file_name = f"temp_{self.global_rank}.jsonl"
        with (self.hparams.default_root_dir / temp_file_name).open(mode="w") as f:
            for result_batch in normalized_hyp_refs:
                print(json.dumps(result_batch), file=f)    
        # sync process
        logger.info(f"Rank{self.global_rank} is waiting..")
        self.trainer.strategy.barrier()

        if self.trainer.is_global_zero:
            now = datetime.today().strftime("%Y%m%d%H%M%S")
            video_id_set = set()
            gatherd_output = []
            for i in range(self.trainer.strategy.world_size):
                temp_file_name = f"temp_{i}.jsonl"
                with (self.hparams.default_root_dir / temp_file_name).open(mode="r") as f_temp:
                    for data in map(json.loads, f_temp):
                        if data["video_id"] in video_id_set:
                            continue
                        else:
                            video_id_set.add(data["video_id"])
                        gatherd_output.append(data)
                            
            result_josn_path = self.hparams.log_dir / f"generation_result_{now}.json"
            with result_josn_path.open(mode="w") as f:  
                json.dump(
                    {
                        "date": now,
                        "result": gatherd_output,
                    },
                    f,
                    indent=4,
                )
            self.logger.experiment.save(str(result_josn_path))
            logger.info(f"Save result to \"{result_josn_path}\"")
            
            # Calc BLEU score
            if self.hparams.joint_evaluation:
                hyp_texts = [
                    " ".join(instance["hyp_texts"])
                    for instance in gatherd_output
                ]
                ref_texts = [
                    " ".join(instance["ref_texts"])
                    for instance in gatherd_output
                ]
            else:
                hyp_texts = sum(
                    (instance["hyp_texts"] for instance in gatherd_output),
                    start=[]
                )
                ref_texts = sum(
                    (instance["ref_texts"] for instance in gatherd_output),
                    start=[]
                )

            
                
            bleu = BLEU(
                tokenize=self.hparams.bleu_tokenizer,
            )
            assert len(hyp_texts) == len(ref_texts), f"{len(hyp_texts)} != {len(ref_texts)}"
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
                (instance["hyp_frame_idxs"] for instance in gatherd_output),
                start=[]
            )
            ref_frame_idxs = sum(
                (instance["ref_frame_idxs"] for instance in gatherd_output),
                start=[]
            )
            frame_accuracy = accuracy_score(hyp_frame_idxs, ref_frame_idxs)
            
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
            for data in gatherd_output[:self.hparams.num_wandb_upload]:
                table_data.append(
                    [
                        data["video_id"],
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
            
