from pathlib import Path
import random

from logzero import logger
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
from more_itertools import windowed


"""                
instance = {
    "labels": labels,
    "video_feature_path": feature_path,
    "video_id": video_id,
    "frame_idxs": [i for c, i in sampled_caption_key_frame_pairs],
}
"""


class DataCollator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.float_dtype = torch.half if self.args.apex else torch.float

    @torch.no_grad()
    def __call__(self, batch_source):
        batch = {k:[] for k in batch_source[0].keys()}
        for instance in batch_source:
            for k, v in instance.items():
                batch[k].append(v)
        
        # decoder padding must be -100 to ignore cross entropy loss
        batch["labels"] = pad_sequence(
            batch["labels"],
            batch_first=True,
            padding_value=-100,
        )
        batch["inputs_embeds"], batch["attention_mask"] = self.make_input_embeds(
            batch["video_feature_path"]
        )
        batch.pop("video_feature_path")
        return batch


    def make_input_embeds(self, video_feature_paths):
        original_features = []
        for path in video_feature_paths:
            feature = torch.tensor(
                np.load(path)[:self.tokenizer.model_max_length],
                dtype=torch.float32,
            )
            original_features.append(feature)
        
        embed_size = original_features[0].size(1)
        max_num_embeddings = max(feature.size(0) for feature in original_features)
        
        attention_masks = []
        padded_features = []
        for feature in original_features:
            num_embeddings = feature.size(0)
            if num_embeddings < max_num_embeddings:
                padding_length = max_num_embeddings - num_embeddings
                feature = torch.cat(
                    (
                        feature,
                        torch.zeros(padding_length, embed_size, dtype=torch.float32)
                    ),
                    dim=0,
                ).unsqueeze(0)
            else:
                feature = feature.unsqueeze(0)
                padding_length = 0
                
            attention_masks.append(
                torch.cat(
                    (
                        torch.ones(num_embeddings, dtype=torch.int64),
                        torch.zeros(padding_length, dtype=torch.int64),
                    )
                ).view(1, -1)
            )
            padded_features.append(feature)
            
        return (
            torch.cat(padded_features, dim=0).to(self.float_dtype),
            torch.cat(attention_masks, dim=0),
        )



"""                
instance = {
    "labels": labels,
    "video_feature_path": feature_path,
    "video_id": video_id,
    "frame_idxs": [i for c, i in sampled_caption_key_frame_pairs],
}
"""
class RerankDataCollator(DataCollator):
    @torch.no_grad()
    def __call__(self, batch_source):
        batch = {k:[] for k in batch_source[0].keys()}
        for instance in batch_source:
            for k, v in instance.items():
                batch[k].append(v)

        batch["inputs_embeds"], batch["attention_mask"] = self.make_input_embeds(
            batch["video_feature_path"]
        )
        batch.pop("video_feature_path")

        if self.args.apex:
            batch["inputs_embeds"] = batch["inputs_embeds"].to(torch.half)
            
            
        if self.args.multi_gold_probability_distribution_labels:
            assert "anotated_time_idxs" in batch
            label_distributions = one_hot(
                pad_sequence(
                    batch["labels"],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                ),
                num_classes=self.tokenizer.vocab_size + self.args.model_max_length,
            ).float()

            
            # Replace the label distribution with the frame distribution
            for i, (frame_idxs, anotated_time_idxs, frame_locations) in enumerate(
                    zip(
                        batch["frame_idxs"],
                        batch["anotated_time_idxs"],
                        batch["frame_locations"],
                    )
            ):
                assert len(frame_idxs) == len(anotated_time_idxs) == len(frame_locations)
                for gold_idx, other_gold_idxs, loc in zip(
                        frame_idxs.tolist(),
                        anotated_time_idxs,
                        frame_locations.tolist(),
                ):
                    frame_gold_labels = torch.tensor(
                        [gold_idx] + other_gold_idxs.tolist(),
                        dtype=torch.long,
                    )
                    frame_distribution = one_hot(
                        frame_gold_labels,
                        num_classes=self.args.model_max_length,
                    ).sum(dim=0).float() / len(frame_gold_labels)
                    
                    label_distributions[i, loc] = torch.cat(
                        (
                            torch.zeros(
                                self.tokenizer.vocab_size,
                                dtype=frame_distribution.dtype,
                            ),
                            frame_distribution,
                        )
                    )
            batch.pop("anotated_time_idxs")
            batch["label_distribution"] = label_distributions
            

        batch["labels"] = pad_sequence(
            batch["labels"],
            batch_first=True,
            padding_value=-100,
        )
        batch["frame_idxs"] = pad_sequence(
            batch["frame_idxs"],
            batch_first=True,
            padding_value=batch["inputs_embeds"].size(1),
        )
        # decoder padding must be -100 to ignore cross entropy loss
        batch["frame_locations"] = pad_sequence(
            batch["frame_locations"],
            batch_first=True,
            padding_value=batch["labels"].size(-1),
        )
        
        # breakpoint()
        if "anotated_time_idxs" in batch:
            max_num_caption = batch["frame_locations"].size(-1)
            max_num_annotated_index = max(
                (
                    len(idxs)
                    for idx_tensors in batch["anotated_time_idxs"]
                    for idxs in idx_tensors
                )
            )
            anotated_time_idxs = []
            for idx_tensors in batch["anotated_time_idxs"]:
                padded_tensor = torch.full(
                    (max_num_caption, max_num_annotated_index),
                    batch["inputs_embeds"].size(1),
                    dtype=torch.long
                )
                for i, idxs in enumerate(idx_tensors):
                    padded_tensor[i, :len(idxs)] = idxs
                    anotated_time_idxs.append(padded_tensor)
            batch["anotated_time_idxs"] = torch.stack(anotated_time_idxs)
            
        return batch
    
class RerankInferenceDataCollator(DataCollator):
    @torch.no_grad()
    def __call__(self, batch_source):
        batch = {k:[] for k in batch_source[0].keys()}
        for instance in batch_source:
            for k, v in instance.items():
                batch[k].append(v)

        batch["inputs_embeds"], batch["attention_mask"] = self.make_input_embeds(
            batch["video_feature_path"]
        )
        batch.pop("video_feature_path")

        if self.args.apex:
            batch["inputs_embeds"] = batch["inputs_embeds"].to(torch.half)

        max_num_frame = max(c.size(0) for c in batch["caption_ids"])
        max_caption_ids_length = max(c.size(1) for c in batch["caption_ids"])

        paded_caption_ids_list = []
        paded_attention_mask_list = []
        for caption_ids, caption_attention_mask in zip(
                batch["caption_ids"],
                batch["caption_attention_mask"],
        ):
            paded_caption_ids = torch.full(
                (max_num_frame, max_caption_ids_length),
                self.tokenizer.pad_token_id,
                dtype=caption_ids.dtype,
            )
            paded_caption_ids[
                :caption_ids.size(0), :caption_ids.size(1)
            ] = caption_ids
            paded_caption_ids_list.append(paded_caption_ids)
            
            paded_attention_mask = torch.full(
                (max_num_frame, max_caption_ids_length),
                0,
                dtype=caption_attention_mask.dtype,
            )
            paded_attention_mask[
                :caption_attention_mask.size(0), :caption_attention_mask.size(1)
            ] = caption_attention_mask
            paded_attention_mask_list.append(paded_attention_mask)
            
        # bsz * max_num_frame * max_seq_len
        batch["caption_ids"] = torch.stack(paded_caption_ids_list, dim=0)
        # bsz * max_num_frame * max_seq_len
        batch["caption_attention_mask"] = torch.stack(paded_attention_mask_list, dim=0)
        return batch


class RerankPsedoVideoDataCollator(DataCollator):
    @torch.no_grad()
    def __call__(self, batch_source):
        batch = {k:[] for k in batch_source[0].keys()}
        for instance in batch_source:
            for k, v in instance.items():
                batch[k].append(v)

        
        batch["inputs_embeds"], batch["frame_idxs"], segumet_ranges = self.make_input_embeds_and_frame_idxs(batch["feature_paths"])
        batch["attention_mask"] = torch.ones(
            batch["inputs_embeds"].size()[:-1],
            dtype=torch.long,
        )
        
        if self.args.apex:
            batch["inputs_embeds"] = batch["inputs_embeds"].to(torch.half)
            
        if self.args.multi_gold_probability_distribution_labels:
            label_distributions = one_hot(
                pad_sequence(
                    batch["labels"],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                ),
                num_classes=self.tokenizer.vocab_size + self.args.model_max_length,
            ).float()
            
            # Replace the label distribution with the frame distribution
            for i, (segment, frame_locations) in enumerate(
                    zip(
                        segumet_ranges,
                        batch["frame_locations"],
                    )
            ):
                for (seg_start, seg_end), loc in zip(
                        windowed(segment, n=2),
                        frame_locations.tolist(),
                ):

                    frame_gold_labels = torch.arange(
                        seg_start,
                        seg_end,
                        dtype=torch.long,
                    )   
                    frame_distribution = one_hot(
                        frame_gold_labels,
                        num_classes=self.args.model_max_length,
                    ).sum(dim=0).float() / len(frame_gold_labels)
                    
                    label_distributions[i, loc] = torch.cat(
                        (
                            torch.zeros(
                                self.tokenizer.vocab_size,
                                dtype=frame_distribution.dtype,
                            ),
                            frame_distribution,
                        )
                    )
            batch["label_distribution"] = label_distributions
            
        # decoder padding must be -100 to ignore cross entropy loss
        batch["labels"] = pad_sequence(
            batch["labels"],
            batch_first=True,
            padding_value=-100,
        )
        batch["frame_locations"] = pad_sequence(
            batch["frame_locations"],
            batch_first=True,
            padding_value=batch["labels"].size(-1),
        )        
        return batch


    def make_input_embeds_and_frame_idxs(self, feature_paths):
        frame_idxs = []
        inputs_embeds = []
        segumet_ranges = []
        for paths in feature_paths:
            source_features = [np.load(path) for path in paths]
            segement_range = random.sample(
                range(1, self.tokenizer.model_max_length),
                k=self.args.num_key_frame - 1
            )
            segement_range.sort()
            segement_range.insert(0, 0)
            segement_range.append(self.tokenizer.model_max_length)

            instance_frame_idxs = []
            embeds = []
            for (seg_start_idx, seg_end_idx), feature in zip(
                    windowed(segement_range, n=2),
                    source_features,
            ):
                embeds.append(
                    feature[None, :].repeat(seg_end_idx - seg_start_idx, axis=0)
                )
                instance_frame_idxs.append(
                    random.randrange(seg_start_idx, seg_end_idx)
                )
            
            assert sum(len(vecs) for vecs in embeds) == self.tokenizer.model_max_length
            inputs_embeds.append(np.concatenate(embeds).tolist())
            frame_idxs.append(instance_frame_idxs)
            segumet_ranges.append(segement_range)
            
        return (
            torch.tensor(inputs_embeds, dtype=torch.float),
            torch.tensor(frame_idxs, dtype=torch.long),
            segumet_ranges,
        )
        
                
            
        










        
