from itertools import accumulate
from pathlib import Path
import json
from datetime import datetime

from logzero import logger
import torch
from torch.nn.functional import one_hot
import wandb
from more_itertools import divide

from .video_reranker_t5_pl import Video2TextRerankerT5PL

class Video2TextRerankerT5InferencePL(Video2TextRerankerT5PL):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Video2TextRerankerT5PL.add_model_specific_args(parent_parser)
        parser.add_argument("--seq_beam", default=1, type=int, required=True)
        parser.add_argument(
            "--caption_json_file_path",
            help="Specify caption file path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--frame_score_weight",
            help="Specify frame score weight",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--ranking_score",
            help="Specify ranking_score mode",
            action="store_true"
        )
        parser.add_argument(
            "--normalized_score",
            help="Specify ranking_score mode",
            action="store_true"
        )
        parser.add_argument(
            "--ranking_score_epsilon",
            help="Specify ranking epsilon",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--normalized_score_epsilon",
            help="Specify ranking epsilon",
            type=float,
            default=0.5,
        )

        return parser


    def _post_init(self):
        super()._post_init()
        self.bos_token_id = self.tokenizer.pad_token_id if self.tokenizer.bos_token_id is None else self.tokenizer.bos_token_id
        assert 0.0 <= self.hparams.ranking_score_epsilon <= 1.0
        
    def forward(self, batch, encoder_outputs=None):
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
        batch["labels"] = self.overwrite_labels_with_frame_token_id(
            batch,
            frame_loc_one_hot=frame_loc_one_hot,
        )
        output = self.model(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            decoder_inputs_embeds=batch["decoder_inputs_embeds"],
            # labels=batch["labels"],
            frame_location_hot = frame_loc_one_hot.sum(
                dim=1
            ) if self.hparams.with_discrete_gate else None,
            encoder_outputs=encoder_outputs,
            output_hidden_states=False,
            output_attentions=False,
        )
        return output
    
    def training_step(self, batch, batch_idx=None):
        raise NotImplementedError()
        
    def validation_step(self, batch, batch_idx=None):
        raise NotImplementedError()
    
    """
    batch = {
        "video_ids": bsz
        "frame_paths": bsz * max_num_frame,
        "inputs_embeds": bsz * max_num_frame * d,
        "attention_mask": bsz * max_frame_length,
        "caption_ids": , # bsz * max_num_frame * max_seq_len # bos_token + eos_tokenをつける
        "caption_attention_mask": , # bsz * max_num_frame * max_seq_len
    }
    """
    def generate_beam_search_batch(self, batch, prefix=None):

        if prefix is None:
            for caption_ids, caption_attention_mask, loc_idx in zip(
                    batch["caption_ids"].permute(1, 0, 2),
                    batch["caption_attention_mask"].permute(1, 0, 2),
                    range(batch["caption_ids"].size(1)),
            ):
                # print(loc_idx)
                labels = caption_ids
                labels[labels==self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
                frame_idxs = torch.full(
                    (
                        caption_ids.size(0),
                        1,
                    ),
                    loc_idx,
                    dtype=torch.long,
                    device=self.device,
                )
                frame_locations = torch.zeros(
                    (caption_ids.size(0), 1),
                    dtype=torch.long,
                    device=self.device,
                )                        
                       
                yield {
                    "inputs_embeds": batch["inputs_embeds"],
                    "attention_mask": batch["attention_mask"],
                    "labels": labels,
                    "labels_attention_mask": caption_attention_mask,
                    "frame_idxs": frame_idxs,
                    "frame_locations": frame_locations,
                    "beam_idx": 0,
                    "loc_idx": loc_idx,
                }
        else:
            # prefix_frame_idxs: bsz * prefix_length
            for beam_idx, prefix_frame_idxs in enumerate(prefix.permute(1, 0, 2)):
                one_hot_prefix_frame_idxs = one_hot(
                    prefix_frame_idxs,
                    num_classes=batch["caption_ids"].size(1),
                )
                last_prefixs_idxs = prefix_frame_idxs[:, -1].min().item() # bsz
                
                prefix_text_ids = torch.einsum(
                    "bfs, bpf->bps",
                    batch["caption_ids"].to(torch.float),
                    one_hot_prefix_frame_idxs.to(torch.float),
                ) # bsz * prefix_length * max_sequence
                # from float to int
                prefix_text_ids = (prefix_text_ids + 0.5).to(batch["caption_ids"].dtype)
                prefix_attention_mask = torch.einsum(
                    "bfs, bpf->bps",
                    batch["caption_attention_mask"].to(torch.float),
                    one_hot_prefix_frame_idxs.to(torch.float),
                ).to(batch["caption_attention_mask"].dtype) # bsz * prefix_length * max_sequence
                # from float to int
                prefix_attention_mask = (prefix_attention_mask + 0.5).to(
                    batch["caption_attention_mask"].dtype
                )
                
                flattened_prefix_text_ids = torch.masked_select(
                    prefix_text_ids,
                    prefix_attention_mask.bool(),
                )
                prefix_caption_lengths = torch.sum(
                    torch.sum(
                        prefix_attention_mask,
                        dim=-1
                    ),
                    dim=-1
                )

                prefix_text_ids[prefix_text_ids==self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
                prefix_caption_tensors = list(
                    map(
                        list,
                        divide(
                            prefix_text_ids.size(0),
                            torch.split(
                                flattened_prefix_text_ids,
                                tuple(
                                    torch.sum(prefix_attention_mask, dim=-1).view(-1)
                                ),
                            ),
                        )
                    )
                ) # bsz * prefix_length * (*)
                # frame_locations = []
                # for tensor_list in prefix_caption_tensors:
                #     frame_locations.append(
                #         torch.tensor(
                #             list(
                #                 accumulate(
                #                     tensor_list,
                #                     lambda total, element: len(element) + total,
                #                     initial=0,
                #                 )
                #             ),
                #             dtype=torch.long,
                #             device=self.device,
                #         )
                #     )
                # frame_locations = torch.stack(frame_locations, dim=0)

                frame_locations = []
                for tensor_list in prefix_caption_tensors:
                    frame_locations.append(
                        list(
                            accumulate(
                                tensor_list,
                                lambda total, element: len(element) + total,
                                initial=0,
                            )
                        )
                    )
                frame_locations = torch.tensor(
                    frame_locations,
                    dtype=torch.long,
                    device=self.device,
                )
                
                # last_prefixs_idxs + 1で, 必要のないforwardは避ける
                for caption_ids, caption_attention_mask, loc_idx in zip(
                        batch["caption_ids"].permute(1, 0, 2)[last_prefixs_idxs + 1:],
                        batch["caption_attention_mask"].permute(1, 0, 2)[last_prefixs_idxs + 1:],
                        range(last_prefixs_idxs + 1, batch["caption_ids"].size(1)),
                ):
                    flattened_new_caption_ids = torch.masked_select(
                        caption_ids,
                        caption_attention_mask.bool(),
                    )
                    new_caption_tensors = list(
                        map(
                            list,
                            divide(
                                caption_ids.size(0),
                                torch.split(
                                    flattened_new_caption_ids,
                                    torch.sum(
                                        caption_attention_mask,
                                        dim=-1
                                    ).view(-1).tolist()
                                ),
                            )
                        )
                    ) # bsz * 1 * (*)
                    new_caption_lengths = torch.sum(
                        caption_attention_mask,
                        dim=-1
                    ) # bsz
                    max_label_length, _ = torch.max(
                        prefix_caption_lengths + new_caption_lengths,
                        dim=-1,
                    )
                    max_label_length = max_label_length.cpu()
                    labels = torch.full(
                        (
                            caption_ids.size(0),
                            min(max_label_length, self.model.config.n_positions),
                        ),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long,
                        device=self.device,
                    )
                    labels_attention_mask = torch.zeros_like(
                        labels,
                        dtype=torch.long,
                        device=self.device,
                    )
                    for i, prefix_captions, new_captions in zip(
                            range(len(prefix_caption_tensors)),
                            prefix_caption_tensors,
                            new_caption_tensors,
                    ):
                        insert_label = torch.cat(
                            prefix_captions + new_captions,
                            dim=-1
                        )[:labels.size(-1)]
                        labels[i, :insert_label.size(0)] = insert_label
                        labels_attention_mask[i, :insert_label.size(0)] = 1

                    if prefix.size(-1) < self.hparams.num_key_frame - 1:
                        labels[labels==self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
                    frame_idxs = torch.cat(
                        (
                            prefix_frame_idxs,
                            torch.full(
                                (
                                    prefix_frame_idxs.size(0),
                                    1,
                                ),
                                loc_idx,
                                dtype=torch.long,
                                device=prefix_frame_idxs.device,
                            ),
                        ),
                        dim=-1,
                    )
                    out_of_max_length_mask = (frame_locations >= labels.size(-1))
                    frame_locations[out_of_max_length_mask] = labels.size(-1)
                    frame_idxs[out_of_max_length_mask] = batch["inputs_embeds"].size(1)
                    
                    out_of_num_max_frame = (frame_idxs >= batch["attention_mask"].sum(dim=-1)[:, None].expand(-1, frame_idxs.size(-1)))
                    frame_idxs[out_of_num_max_frame] = batch["inputs_embeds"].size(1)
                    frame_locations[out_of_num_max_frame] = labels.size(-1)
                    
                    
                    yield {
                        "inputs_embeds": batch["inputs_embeds"],
                        "attention_mask": batch["attention_mask"],
                        "labels": labels,
                        "labels_attention_mask": labels_attention_mask,
                        "frame_idxs": frame_idxs,
                        "frame_locations": frame_locations,
                        "beam_idx": beam_idx,
                        "loc_idx": loc_idx,
                    }
        

        
        
    

    
    def test_step(self, batch, batch_idx=None):
        video_ids = batch.pop("video_id")
        
        # Get encoder output
        if self.model.encoder_embedding_projection is not None:
            inputs_embeds = self.model.encoder_embedding_projection(
                batch["inputs_embeds"]
            )
        else:
            inputs_embeds = batch["inputs_embeds"]
        
        encoder_outputs = self.model.encoder(
            attention_mask=batch["attention_mask"],
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        
        bsz = inputs_embeds.size(0)
        prefix = None # None or bsz * beam_width * prefix_length
        for i in range(self.hparams.num_key_frame):
            
            if self.hparams.ranking_score or self.hparams.normalized_score:
                scores = torch.full(
                    (
                        inputs_embeds.size(1),
                        1 if prefix is None else prefix.size(1),
                        bsz,
                        2,
                    ),
                    float("-inf"),
                    dtype=inputs_embeds.dtype,
                    device=self.device,
                ) # num_frame * beam_width * bsz * 2(frame_score, caption_score)
            else:
                scores = torch.full(
                    (
                        inputs_embeds.size(1),
                        1 if prefix is None else prefix.size(1),
                        bsz,
                    ),
                    float("-inf"),
                    dtype=inputs_embeds.dtype,
                    device=self.device,
                ) # num_frame * beam_width * bsz
                
            for inner_batch in self.generate_beam_search_batch(batch, prefix=prefix):    
                output = self(
                    inner_batch,
                    encoder_outputs,
                )
                one_hot_label = one_hot(
                    inner_batch["labels"],
                    num_classes=self.model.config.n_positions + self.tokenizer.vocab_size
                )
                token_score = torch.sum(output.logits * one_hot_label, dim=-1)
                att_count = torch.sum(
                    inner_batch["labels_attention_mask"],
                    dim=-1
                )
                frame_loc_hot = one_hot(
                    inner_batch["frame_locations"],
                    num_classes=inner_batch["labels"].size(-1) + 1
                )[:, :, :-1].sum(dim=-2).bool()
                
                if self.hparams.ranking_score or self.hparams.normalized_score:
                    frame_att_count = torch.sum(frame_loc_hot.long(), dim=-1)
                    frame_score = torch.sum(
                        token_score * frame_loc_hot.to(token_score.dtype),
                        dim=-1
                    ) / frame_att_count
                    caption_attention = torch.logical_not(frame_loc_hot).to(
                        token_score.dtype
                    )  * inner_batch["labels_attention_mask"]
                    caption_att_count = torch.sum(caption_attention, dim=-1)
                    caption_score = torch.sum(
                        token_score * caption_attention,
                        dim=-1
                    ) / caption_att_count
                    caption_score[caption_att_count == 0] = float("-inf")
                    seq_score = torch.stack((frame_score, caption_score)).T
                else:
                    # Apply frame score weight
                    score_weight = torch.ones_like(token_score)
                    score_weight[frame_loc_hot] = self.hparams.frame_score_weight
                    seq_score = torch.sum(
                        score_weight * token_score * inner_batch["labels_attention_mask"],
                        dim=-1
                    ) / att_count
                    
                # Replace all padding (no score_attention_mask) input`s score to -inf
                seq_score = torch.nan_to_num(
                    seq_score,
                    nan=float('-inf'),
                    posinf=float('-inf')
                )
                scores[
                    inner_batch["loc_idx"], inner_batch["beam_idx"], :,
                ] = seq_score.float()

            if self.hparams.ranking_score:
                frame_score = scores[:, :, :, 0].permute(2, 1, 0).contiguous()
                # bsz * beam_width * num_frame
                caption_score = scores[:, :, :, 1].permute(2, 1, 0).contiguous()
                # bsz * beam_width * num_frame
                frame_rank_score = (torch.argsort(
                    torch.argsort(
                        frame_score.view(bsz, -1),
                        dim=-1,
                    ),
                    dim=-1,
                ).to(frame_score.dtype) + 1) * self.hparams.ranking_score_epsilon
                caption_rank_score = (torch.argsort(
                    torch.argsort(
                        caption_score.contiguous(),
                        dim=-1,
                    ),
                    dim=-1,
                ).to(caption_score.dtype) + 1) * (1 - self.hparams.ranking_score_epsilon)
                scores = (
                    frame_rank_score.view(
                        *frame_score.size()
                    ) + caption_rank_score.view(
                        *caption_score.size()
                    )
                )
                # scores = scores.permute(2, 1, 0).contiguous()
            elif self.hparams.normalized_score:
                frame_score = scores[:, :, :, 0].permute(2, 1, 0).contiguous()
                # bsz * beam_width * num_frame
                caption_score = scores[:, :, :, 1].permute(2, 1, 0).contiguous()
                # bsz * beam_width * num_frame
                # normalize score to [0, 1] for each beam with min-max normalization
                min_frame_score = frame_score.where(
                    frame_score.isneginf().logical_not(),
                    torch.full_like(frame_score, float("inf"))
                ).view(bsz, -1).min(dim=-1)[0].view(bsz, 1, 1)
                max_frame_score = frame_score.where(
                    frame_score.isposinf().logical_not(),
                    torch.full_like(frame_score, float("-inf"))
                ).view(bsz, -1).max(dim=-1)[0].view(bsz, 1, 1)
                frame_score = (frame_score - min_frame_score) / (max_frame_score - min_frame_score)
                min_caption_score = caption_score.where(
                    caption_score.isneginf().logical_not(),
                    torch.full_like(caption_score, float("inf"))
                ).view(bsz, -1).min(dim=-1)[0].view(bsz, 1, 1)
                max_caption_score = caption_score.where(
                    caption_score.isposinf().logical_not(),
                    torch.full_like(caption_score, float("-inf"))
                ).view(bsz, -1).max(dim=-1)[0].view(bsz, 1, 1)
                caption_score = (caption_score - min_caption_score) / (max_caption_score - min_caption_score)
                scores = frame_score * self.hparams.normalized_score_epsilon + caption_score * (1 - self.hparams.normalized_score_epsilon)
            else:
                scores = scores.permute(2, 1, 0).contiguous() # bsz * beam_width * num_frame
                
                
            
            # Apply mask to enforce selecting after time frame
            time_mask = torch.full_like(scores, float("-inf"))
            num_frames = batch["attention_mask"].sum(dim=-1).cpu()
            remain_key_frames = self.hparams.num_key_frame - (i + 1)
            if prefix is None:
                for batch_idx, num_frame in enumerate(num_frames):
                    time_mask[batch_idx, :, 0:num_frame - remain_key_frames] = 0
            else:
                for batch_idx, (last_prefixs_per_batch, num_frame) in enumerate(
                        zip(
                            prefix[:, :, -1],
                            num_frames,
                        )
                ):
                    for beam_idx, last_frame_idx in enumerate(last_prefixs_per_batch):
                        time_mask[
                            batch_idx,
                            beam_idx,
                            last_frame_idx + 1:num_frame - remain_key_frames
                        ] = 0
            scores = scores + time_mask


            flattened_score = scores.view(bsz, -1)
            topk_score, topk_idxs = torch.topk(
                flattened_score,
                k=self.hparams.seq_beam,
                dim=-1
            )
            topk_idx_hot = torch.sum(
                one_hot(topk_idxs, num_classes=flattened_score.size(-1)),
                dim=-2,
            ).view(*scores.size())
            topk_score_idxs = topk_idx_hot.nonzero()
            # (bsz * topk) * 3(batch_idx, beam_idx, frame_idx))
            
            if prefix is None: # None or bsz * beam_width * prefix_length
                prefix = topk_score_idxs[: ,2].view(
                    bsz,
                    -1
                ).unsqueeze(-1)
            else:
                topk_score_idxs = topk_score_idxs[:, 1:].view(bsz, -1, 2)
                past_prefix = torch.einsum(
                    "bwp, btw->btp",
                    prefix.to(torch.float),
                    one_hot(
                        topk_score_idxs[:, :, 0],
                        num_classes=prefix.size(1),
                    ).to(torch.float)
                )
                # from float to int
                past_prefix = (past_prefix + 0.5).to(prefix.dtype)
                prefix = torch.cat(
                    (
                        past_prefix,
                        topk_score_idxs[:, :, 1, None]
                    ),
                    dim=-1,
                )
            
        return {
            "video_ids": video_ids,
            "n_best": prefix.cpu(),
            "scores": topk_score.cpu(),
        }

    def test_epoch_end(self, outputs):
        temp_file_name = f"temp_{self.global_rank}.jsonl"
        with (self.hparams.default_root_dir / temp_file_name).open(mode="w") as f:
            for result_batch in outputs:
                for video_id, n_best, score in zip(*tuple(result_batch.values())):
                    instance = {
                        "video_id": video_id,
                        "n_best_frame_index": n_best.cpu().tolist(),
                        "score": score.cpu().tolist(),
                    }
                    print(json.dumps(instance), file=f)

        # sync process
        logger.info(f"Rank{self.global_rank} is waiting..")
        self.trainer.strategy.barrier()
        if self.trainer.is_global_zero:
            now = datetime.today().strftime("%Y%m%d%H%M%S")
            with (self.hparams.log_dir / f"reranking_result_{now}.json").open(mode="w") as f:
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
                            
            result_josn_path = self.hparams.log_dir / f"reranking_result_{now}.json"
            with result_josn_path.open(mode="w") as f:  
                json.dump(
                    {
                        "date": now,
                        "caption_file_path": str(self.hparams.caption_json_file_path),
                        "result": gatherd_output,
                    },
                    f,
                    indent=4,
                )
            self.logger.experiment.save(str(result_josn_path))
            logger.info(f"Save result to \"{result_josn_path}\"")
            
            with self.hparams.caption_json_file_path.open(mode="r") as f:
                caption_data_json = json.load(f)
                table_data = []
                for data in gatherd_output[:self.hparams.num_wandb_upload]:
                    video_id = data["video_id"]
                    frame_idxs = data["n_best_frame_index"][0]
                    hyp_text = [
                        caption_data_json[video_id]["frames"][i]["caption"]
                        for i in frame_idxs
                    ]
                    table_data.append(
                        [
                            video_id,
                            hyp_text,
                            frame_idxs,
                            data["score"][0],
                        ]
                    )
                
                self.logger.log_table(
                    key="Predictions",
                    columns=[
                        "video_id",
                        "hyp_text",
                        "hyp_frame_idx",
                        "score",
                    ],
                    data=table_data,
                )
                
