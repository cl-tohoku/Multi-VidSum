from pathlib import Path
import json
import random
from itertools import islice, groupby, chain
import csv
from collections import Counter
import os

import numpy as np
import torch
from logzero import logger
from tqdm.auto import tqdm
from pytorch_lightning.utilities.seed import seed_everything
from transformers import AutoTokenizer
import joblib
from more_itertools import chunked

from utils.joblib_utils import joblib_tqdm
from utils.pickle_file import PickleFileWriter
from utils.joblib_utils import joblib_tqdm



class MetaDataPreProcessor:
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--raw_data_path",
            help="Specify sentenceanno_result_allin.sort_sid.trimed_video_sorted.tsv path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--output_json_path",
            help="Specify output_json_path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--subset_size",
            help="Specify subset size",
            type=int,
        )

    
    def __init__(self, args):
        self.args = args
        
    def __call__(self):
        self.args.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        key_frame_count = []
        
        with self.args.raw_data_path.open(mode="r") as f_in, \
             self.args.output_json_path.open(mode="w") as f_out:
            reader = csv.reader(f_in, delimiter="\t")
            
            for video_name, infos_per_video in tqdm(
                islice(
                    groupby(reader, lambda data: data[1]),
                    self.args.subset_size,
                )
            ):
                infos_per_video = list(infos_per_video)
                infos_per_video.sort(key=lambda x: x[-2])
                
                for split, infos in groupby(infos_per_video, lambda x: x[-2]):
                    infos = list(infos)
                    sent_ids = list(set(data[2] for data in infos))
                    output_json_dict = {
                        "video_name": video_name,
                        "split": split,
                        "movie_length": float(infos[0][-1]),
                        "caption_info": {},
                    }


                    for task_id, video_name, sent_id, num_submit, anotated_time, en_caption, ja_caption, segment_start, segment_end, split, movie_length in infos:
                        if sent_id not in output_json_dict["caption_info"]:
                            output_json_dict["caption_info"][sent_id] = {
                                "caption": en_caption,
                                "segment_start": float(segment_start),
                                "segment_end": float(segment_end),
                                "anotated_times": [],
                            }

                        output_json_dict["caption_info"][sent_id]["anotated_times"].append(
                            float(anotated_time)
                        )

                    print(json.dumps(output_json_dict, ensure_ascii=False), file=f_out)
                    key_frame_count.append(len(output_json_dict["caption_info"]))


class VISTMetaDataPreProcessor(MetaDataPreProcessor):
    @staticmethod
    def add_args(parser):
        MetaDataPreProcessor.add_args(parser)
        parser.add_argument(
            "--image_dir",
            help="Specify image dir",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--feature_dir",
            help="Specify feature dir",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--feature_model_name",
            help="Specify feature model name",
            type=str,
            default="clip"
        )
        
    """
    {
        "input_ids": ,
        "pair_index": ,
    }
    """
    def __call__(self):
        self.args.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_dict = {}
        
        with self.args.raw_data_path.open(mode="r") as f_in, \
             self.args.output_json_path.open(mode="w") as f_out:
            raw_json_dict = json.load(f_in)
            for data in raw_json_dict["annotations"]:
                assert len(data) == 1
                data = data[0]
                story_id = data["story_id"]
                if story_id not in output_json_dict:
                    output_json_dict[story_id] = []
                output_json_dict[story_id].append(
                    {
                        "text": data["original_text"].strip(),
                        "order": int(data["worker_arranged_photo_order"]),
                        "image_path": self.args.image_dir / f"{data['photo_flickr_id']}.jpg",
                        "feature_path": self.args.feature_dir /  f"{data['photo_flickr_id']}_{self.args.feature_model_name}.npy"
                    }
                )
        
        remove_keys = []
        count_of_story_before_filtering = len(output_json_dict)
        logger.info(f"There are {count_of_story_before_filtering} stories before filtering")
        for story_id, data in output_json_dict.items():
            data.sort(key=lambda d:d["order"])
            if any(not d["image_path"].exists() for d in data):
                remove_keys.append(story_id)
                continue
            if any(not d["feature_path"].exists() for d in data):
                remove_keys.append(story_id)
                continue
            
            order_numbers = []
            for d in data:
                d["image_path"] = str(d["image_path"])
                d["feature_path"] = str(d["feature_path"])
                order_numbers.append(d["order"])
            assert list(range(len(data))) == order_numbers
            
        for story_id in remove_keys:
            logger.info(f"Data of story id {story_id} was removed")
            del output_json_dict[story_id]
        
        with self.args.output_json_path.open(mode="w") as f_out:
            json.dump(output_json_dict, f_out, indent=4)
        
        logger.info(
            "\n" + 
            json.dumps(
                {
                    "Number of stories": len(output_json_dict),
                    "Number of stories before filtering": count_of_story_before_filtering,
                },
                indent=4,
            )
        )
        
class PicklizePreProcessor:
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--meta_data_path",
            help="Specify meta data path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--output_dir",
            help="Specify output_dir",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--feature_dir",
            help="Specify output_json_path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--feature_model_name",
            help="Specify model name to create feature",
            type=str,
            default="clip",
        )
        parser.add_argument(
            "--tokenizer_name_or_path",
            help="Specify tokenizer name or path",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--split",
            help="Specify data split",
            type=str,
            nargs="*",
        )
        parser.add_argument(
            "--subset_size",
            help="Specify subset size",
            type=int,
        )
        parser.add_argument(
            "--seed",
            help="Specify seed",
            type=int,
            default="42",
        )
        parser.add_argument(
            "--num_key_frame",
            help="Specify minimum number of key frame. if not specified, all anotated frame will be used.",
            type=int,
        )
        parser.add_argument(
            "--num_sampling",
            help="Specify number of instance per video.",
            type=int,
            required=True,
        )
        parser.add_argument(
            "--discretized_frequency",
            help="Specify video discretized frequency",
            type=float,
            required=True,
        )
        parser.add_argument(
            "--model_max_length",
            help="Specify number of max input tokens in encoder.",
            type=int,
        )
        parser.add_argument(
            "--parallel",
            action='store_true'
        )

        
    def __init__(self, args):
        self.args = args
        os.environ["TOKENIZERS_PARALLELISM"] = str(not args.parallel).lower()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path)
        
        if self.args.model_max_length is not None:
            self.tokenizer.model_max_length = self.args.model_max_length
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        self.bos_token_id = self.tokenizer.pad_token_id if self.tokenizer.bos_token_id is None else self.tokenizer.bos_token_id
            
    def get_nearest_value_index(self, t, candidates):
        return int(np.abs(candidates - t).argmin())
    
    
    def __call__(self):
        assert self.args.feature_dir.exists()
        basic_info_path = self.args.output_dir / "basic_info.json"
        tokenized_data_file_path = self.args.output_dir / "tokenized_data.pkl"
        
        with self.args.meta_data_path.open(mode="r") as f, \
             PickleFileWriter(tokenized_data_file_path) as pickle_writer:
            count = 0
            
            if self.args.split is not None:
                pbar = tqdm(
                    filter(
                        lambda d: d["split"] in self.args.split,
                        map(lambda l: json.loads(l), f),
                    ),
                )
            else:
                pbar = tqdm(map(lambda l: json.loads(l), f))
                
            for json_dict in pbar:
                video_id = json_dict["video_name"][2:]
                feature_path = self.args.feature_dir / f"{video_id}_clip.npy"
                try:
                    feature = np.load(feature_path) # Number of frame x feature size
                except FileNotFoundError:
                    logger.info(f"{feature_path} dose not exist..")
                    continue

                # Truncation
                feature = feature[:self.tokenizer.model_max_length]
                frame_times = np.linspace(
                    0,
                    self.args.discretized_frequency * feature.shape[0],
                    feature.shape[0],
                    endpoint=False,
                )
                last_frame_time = frame_times[-1]
                remove_keys = []
                for k, v in json_dict["caption_info"].items():
                    if all(
                            t > last_frame_time + self.args.discretized_frequency
                            for t in v["anotated_times"]
                    ):
                        remove_keys.append(k)
                
                for k in remove_keys:
                    json_dict["caption_info"].pop(k)

                
                if len(json_dict["caption_info"]) == 0:
                    logger.info(
                        f"All captions in {json_dict['video_name']} was truncated, skip!"
                    )
                    continue
                
                
                if self.args.num_key_frame is not None:
                    if len(json_dict["caption_info"]) < self.args.num_key_frame:
                        logger.info(
                            f"Only {len(json_dict['caption_info'])} key frame exist. Not enough keyframes.."
                        )
                        continue



                if self.args.similarity_based_truncation:
                    feature_path = self.similarity_based_truncation(feature_path)
                    
            
                for _ in range(self.args.num_sampling):
                    sampled_caption_key_frame_pairs = []
                    
                    if self.args.num_key_frame is None:
                        caption_infos = list(json_dict["caption_info"].values())
                    else:
                        caption_infos = random.sample(
                            list(json_dict["caption_info"].values()),
                            self.args.num_key_frame
                        )
                        assert len(caption_infos) == self.args.num_key_frame
                    
                    for v in caption_infos:
                        sampled_caption_key_frame_pairs.append(
                            (
                                v["caption"],
                                self.get_nearest_value_index(
                                    random.choice(v["anotated_times"]),
                                    frame_times,
                                )
                            )
                        )
                    sampled_caption_key_frame_pairs.sort(key=lambda p: p[1])
                
                    label_tensors = []
                    with self.tokenizer.as_target_tokenizer():
                        for caption, frame_idx in sampled_caption_key_frame_pairs:
                            # Remove eos(sep) token
                            tokenized_caption_ids = self.tokenizer(
                                caption,
                                padding=False,
                                truncation=True,
                                return_tensors='pt'
                            ).input_ids[0][:-1]
                            
                            label_tensors.append(
                                torch.tensor(
                                    self.tokenizer.vocab_size + frame_idx,
                                    dtype=tokenized_caption_ids.dtype,
                                ).view(-1)
                            )
                            label_tensors.append(tokenized_caption_ids)
                            label_tensors.append(
                                torch.tensor(
                                    self.bos_token_id,
                                    dtype=tokenized_caption_ids.dtype,
                                ).view(-1)
                            )
                            
                    # Add eos token
                    label_tensors = label_tensors[:-1]
                    label_tensors.append(
                        torch.tensor(
                            self.tokenizer.eos_token_id,
                            dtype=tokenized_caption_ids.dtype,
                        ).view(-1)
                    )
                        
                    
                    # Concat and truncation
                    labels = torch.cat(
                        label_tensors, dim=0
                    )[:self.tokenizer.model_max_length]
                    
                    instance = {
                        "labels": labels,
                        "video_feature_path": feature_path,
                        "video_id": video_id,
                        "frame_idxs": [i for c, i in sampled_caption_key_frame_pairs],
                    }
                    
                    pickle_writer.write(instance)
                    count += 1
                    pbar.set_postfix(count=count)
                    
                    if (self.args.subset_size is not None) and (count == self.args.subset_size):
                        logger.info("Finish!")
                        break
                    
                else:
                    continue
                break

        
        basic_info = {
            "collator": "DataCollator",
            "data_size": count,
            "meta_data_path": str(self.args.meta_data_path),
        }
        with basic_info_path.open(mode="w") as f:
            json.dump(basic_info, f, indent=4)
        print(json.dumps(basic_info, indent=4))
            
            

class PicklizeRerankPreProcessor(PicklizePreProcessor):
    @staticmethod
    def add_args(parser):
        PicklizePreProcessor.add_args(parser)
        parser.add_argument(
            "--use_insufficient_key_frame_movies",
            help="Use similarity based truncation",
            action="store_true",
        )
        parser.add_argument(
            "--post_frame",
            help="Whether to use post frame mode",
            action="store_true",
        )
        parser.add_argument(
            "--multiple_gold_learning",
            help="Whether to use post multiple gold learning",
            action="store_true",
        )

        
    def __init__(self, args):
        super().__init__(args)
        self.frame_dummy_id = 1000000000000

    def make_instances(self, json_dict):
        video_id = json_dict["video_name"][2:]
        feature_path = self.args.feature_dir / f"{video_id}_clip.npy"
        try:
            feature = np.load(feature_path) # Number of frame x feature size
        except FileNotFoundError:
            logger.info(f"{feature_path} dose not exist..")
            return []

        # Truncation
        feature = feature[:self.tokenizer.model_max_length]
        frame_times = np.linspace(
            0,
            self.args.discretized_frequency * feature.shape[0],
            feature.shape[0],
            endpoint=False,
        )
        last_frame_time = frame_times[-1]
        remove_keys = []
        for k, v in json_dict["caption_info"].items():
            if all(
                    t > last_frame_time + self.args.discretized_frequency
                    for t in v["anotated_times"]
            ):
                remove_keys.append(k)

        for k in remove_keys:
            json_dict["caption_info"].pop(k)


        if len(json_dict["caption_info"]) == 0:
            logger.info(
                f"All captions in {json_dict['video_name']} was truncated, skip!"
            )
            return []
        

        if not self.args.use_insufficient_key_frame_movies:
            if len(json_dict["caption_info"]) < self.args.num_key_frame:
                logger.info(
                    f"Only {len(json_dict['caption_info'])} key frame exist. Not enough keyframes.."
                )
                return []

        instances = []
        for _ in range(self.args.num_sampling):
            sampled_caption_key_frame_pairs = []                    
            caption_infos = random.sample(
                list(json_dict["caption_info"].values()),
                min(self.args.num_key_frame, len(json_dict["caption_info"])),
            )

            gold_indices = [
                self.get_nearest_value_index(
                    random.choice(v["anotated_times"]),
                    frame_times,
                )
                for v in caption_infos
            ]
            if self.args.multiple_gold_learning:
                anotated_indexes = []
                for v, gold_idx in zip(caption_infos, gold_indices):
                    sampled_caption_key_frame_pairs.append(
                        (
                            v["caption"].strip(),
                            gold_idx
                        )
                    )
                    anotated_indexes.append(
                        list(
                            set(
                                self.get_nearest_value_index(t, frame_times)
                                for t in v["anotated_times"]
                            ) - {gold_idx}
                        )
                    )
                assert len(sampled_caption_key_frame_pairs) == len(anotated_indexes)
                sampled_caption_key_frame_pairs, anotated_indexes = map(
                    list,
                    zip(
                        *sorted(
                            zip(sampled_caption_key_frame_pairs, anotated_indexes),
                            key=lambda t: t[0][1],
                        )
                    )
                )
                # Remove indexes which is more than next gold index
                for i, ((cap, gold_idx), indexes) in enumerate(
                    zip(
                        sampled_caption_key_frame_pairs[:-1],
                        anotated_indexes[:-1],
                    )
                ):
                    # 1つ先のgoldよりも大きいindexは削除
                    new_indexes = [
                        idx
                        for idx in indexes
                        if idx < sampled_caption_key_frame_pairs[i + 1][1]
                    ]
                    anotated_indexes[i] = torch.tensor(new_indexes, dtype=torch.long)
                anotated_indexes[-1] = torch.tensor(
                    anotated_indexes[-1],
                    dtype=torch.long
                )
                
            else:
                anotated_indexes = None
                for v, gold_idx in zip(caption_infos, gold_indices):
                    sampled_caption_key_frame_pairs.append(
                        (
                            v["caption"].strip(),
                            gold_idx,
                        )
                    )
                sampled_caption_key_frame_pairs.sort(key=lambda p: p[1])


            label_tensors = []
            frame_idxs = []
            frame_locations = []
            if self.args.post_frame:
                next_frame_location = -2
            else:
                next_frame_location = 0

            with self.tokenizer.as_target_tokenizer():
                for caption, frame_idx in sampled_caption_key_frame_pairs:
                    tokenized_caption_ids = self.tokenizer(
                        caption,
                        padding=False,
                        truncation=True,
                        return_tensors='pt'
                    ).input_ids[0][:-1]
                    frame_idxs.append(frame_idx)

                    if self.args.post_frame:
                        label_tensors.append(tokenized_caption_ids)
                        label_tensors.append(
                            torch.tensor(
                                self.tokenizer.pad_token_id,
                                dtype=torch.long,
                            ).view(-1)
                        )
                        label_tensors.append(
                            torch.tensor(
                                self.bos_token_id,
                                dtype=tokenized_caption_ids.dtype,
                            ).view(-1)
                        )
                        next_frame_location += (len(tokenized_caption_ids) + 1 + 1)
                        frame_locations.append(next_frame_location)
                    else:
                        label_tensors.append(
                            torch.tensor(
                                self.tokenizer.pad_token_id,
                                dtype=torch.long,
                            ).view(-1)
                        )
                        frame_locations.append(next_frame_location)
                        label_tensors.append(tokenized_caption_ids)
                        label_tensors.append(
                            torch.tensor(
                                self.bos_token_id,
                                dtype=tokenized_caption_ids.dtype,
                            ).view(-1)
                        )
                        # frame_id + len(tokenized_caption_ids) + bos_token_id
                        next_frame_location += (1 + len(tokenized_caption_ids) + 1)


            label_tensors = label_tensors[:-1]
            label_tensors.append(
                torch.tensor(
                    self.tokenizer.eos_token_id,
                    dtype=tokenized_caption_ids.dtype,
                ).view(-1)
            )

            # Truncate frame informations
            frame_locations = list(
                filter(
                    lambda i: i < self.tokenizer.model_max_length,
                    frame_locations,
                )
            )
            frame_idxs = torch.tensor(
                frame_idxs[:len(frame_locations)],
                dtype=torch.long
            )
            frame_locations = torch.tensor(frame_locations, dtype=torch.long)

            # Concat and truncation
            labels = torch.cat(
                label_tensors, dim=0
            )[:self.tokenizer.model_max_length]
            assert torch.all(
                torch.index_select(
                    labels, 0, frame_locations, 
                ) == self.tokenizer.pad_token_id,
                dim=0,
            )

            instance = {
                "labels": labels,
                "video_feature_path": feature_path,
                "video_id": video_id,
                "frame_idxs": frame_idxs,
                "frame_locations": frame_locations
            }

            if self.args.multiple_gold_learning:
                assert anotated_indexes is not None
                anotated_indexes = anotated_indexes[:len(frame_locations)]
                assert len(frame_locations) == len(anotated_indexes)
                instance["anotated_time_idxs"] = anotated_indexes
            instances.append(instance)
        
        return instances

    
    def __call__(self):
        assert self.args.feature_dir.exists()
        basic_info_path = self.args.output_dir / "basic_info.json"
        tokenized_data_file_path = self.args.output_dir / "tokenized_data.pkl"
        
        with self.args.meta_data_path.open(mode="r") as f:
            count = 0
            if self.args.split is not None:
                meta_data = list(
                    filter(
                        lambda d: d["split"] in self.args.split,
                        map(lambda l: json.loads(l), f),
                    )
                )
            else:
                meta_data = list(map(lambda l: json.loads(l), f))


        with PickleFileWriter(tokenized_data_file_path) as pickle_writer:
            if self.args.parallel:
                with joblib_tqdm(len(meta_data)):
                    instances_list = joblib.Parallel(n_jobs=-1)(
                        joblib.delayed(self.make_instances)(json_dict)
                        for json_dict in meta_data
                    )
                    
                for instance in chain.from_iterable(instances_list):
                    pickle_writer.write(instance)
                    count += 1
                    if (self.args.subset_size is not None) and (count == self.args.subset_size):
                        logger.info("Finish!")
                        break
            else:
                for json_dict in tqdm(meta_data):
                    for instance in self.make_instances(json_dict):
                        pickle_writer.write(instance)
                        count += 1
                        if (self.args.subset_size is not None) and (count == self.args.subset_size):
                            logger.info("Finish!")
                            break
                    else:
                        continue
                    break
        
        basic_info = {
            "collator": "RerankDataCollator",
            "data_size": count,
            "meta_data_path": str(self.args.meta_data_path),
        }
        with basic_info_path.open(mode="w") as f:
            json.dump(basic_info, f, indent=4)
        logger.info(f"Conclusion: {json.dumps(basic_info, indent=4)}")
        


class PicklizeRerankInferencePreProcessor(PicklizePreProcessor):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--caption_json_file_path",
            help="Specify caption file path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--output_dir",
            help="Specify output_dir",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--feature_dir",
            help="Specify output_json_path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--feature_model_name",
            help="Specify model name to create feature",
            type=str,
            default="clip",
        )
        parser.add_argument(
            "--tokenizer_name_or_path",
            help="Specify tokenizer name or path",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--subset_size",
            help="Specify subset size",
            type=int,
        )
        parser.add_argument(
            "--seed",
            help="Specify seed",
            type=int,
            default="42",
        )
        parser.add_argument(
            "--model_max_length",
            help="Specify number of max input tokens in encoder.",
            type=int,
        )
        parser.add_argument(
            "--parallel",
            action='store_true'
        )
        parser.add_argument(
            "--post_frame",
            help="Whether to use post frame mode",
            action="store_true",
        )
        

    def make_instance(self, video_id, data):
        feature_path = self.args.feature_dir / f"{video_id}_{self.args.feature_model_name}.npy"
        if self.args.post_frame:
            captions = [
                d["caption"].strip() + " " + self.tokenizer.pad_token
                for d in data["frames"][:self.tokenizer.model_max_length]
            ] 
            tokenizer_output = self.tokenizer(
                captions,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            # text ids + pad_token id + eos_token id
            caption_ids = tokenizer_output.input_ids[:, :self.tokenizer.model_max_length]
            caption_attention_mask = tokenizer_output.attention_mask[:, :self.tokenizer.model_max_length]
        else:
            captions = [
                d["caption"].strip()
                for d in data["frames"][:self.tokenizer.model_max_length]
            ] 
            tokenizer_output = self.tokenizer(
                captions,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            # bos_token id + text ids + eos_token id
            caption_ids = torch.cat(
                (
                    torch.full(
                        (tokenizer_output.input_ids.size(0), 1),
                        self.bos_token_id,
                        dtype=tokenizer_output.input_ids.dtype
                    ),
                    tokenizer_output.input_ids,
                ),
                dim=-1,
            )[:, :self.tokenizer.model_max_length]

            caption_attention_mask = torch.cat(
                (
                    torch.full(
                        (tokenizer_output.attention_mask.size(0), 1),
                        1,
                        dtype=tokenizer_output.attention_mask.dtype
                    ),
                    tokenizer_output.attention_mask,
                ),
                dim=-1,
            )[:, :self.tokenizer.model_max_length]
        
        instance = {
            "video_id": video_id,
            "video_feature_path": feature_path,
            "frame_paths": [d["path"] for d in data["frames"][:len(captions)]],
            "caption_ids": caption_ids,
            "caption_attention_mask": caption_attention_mask,
        }
        assert len(instance["frame_paths"]) == len(instance["caption_ids"]) == len(instance["caption_attention_mask"])
        return instance
        
    @torch.no_grad()
    def __call__(self):
        assert self.args.feature_dir.exists()
        basic_info_path = self.args.output_dir / "basic_info.json"
        tokenized_data_file_path = self.args.output_dir / "tokenized_data.pkl"
        with self.args.caption_json_file_path.open(mode="r") as f:
            caption_data = json.load(f)

        if self.args.parallel:
            with joblib_tqdm(len(caption_data)):
                dataset = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(self.make_instance)(video_id, data)
                    for video_id, data in caption_data.items()
                )

            with PickleFileWriter(tokenized_data_file_path) as pickle_writer:
                for instance in tqdm(dataset):
                    pickle_writer.write(instance)
            
        else:
            with PickleFileWriter(tokenized_data_file_path) as pickle_writer:
                for video_id, data in tqdm(caption_data.items(), total=len(caption_data)):
                    pickle_writer.write(self.make_instance(video_id, data))
                    
        
        basic_info = {
            "collator": "RerankInferenceDataCollator",
            "data_size": len(caption_data),
            "caption_json_file_path": str(self.args.caption_json_file_path),
        }
        with basic_info_path.open(mode="w") as f:
            json.dump(basic_info, f, indent=4)
        logger.info(f"Conclusion: {json.dumps(basic_info, indent=4)}")
        




class PicklizeRerankCocoPseudoVideoPreProcessor(PicklizeRerankPreProcessor):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--caption_file_path",
            help="Specify caption_file_path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--output_dir",
            help="Specify output_dir",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--feature_dir",
            help="Specify output_json_path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--feature_model_name",
            help="Specify model name to create feature",
            type=str,
            default="clip",
        )
        parser.add_argument(
            "--tokenizer_name_or_path",
            help="Specify tokenizer name or path",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--num_instances",
            help="Number of instances.",
            type=int,
            required=True,
        )
        parser.add_argument(
            "--subset_size",
            help="Specify subset size",
            type=int,
        )
        parser.add_argument(
            "--seed",
            help="Specify seed",
            type=int,
            default="42",
        )
        parser.add_argument(
            "--num_key_frame",
            help="Specify minimum number of key frame. if not specified, all anotated frame will be used.",
            type=int,
        )
        parser.add_argument(
            "--model_max_length",
            help="Specify number of max input tokens in encoder.",
            type=int,
        )
        parser.add_argument(
            "--post_frame",
            help="Whether to use post frame mode",
            action="store_true",
        )
        parser.add_argument(
            "--parallel",
            action='store_true'
        )

    @torch.no_grad()
    def __call__(self):
        assert self.args.feature_dir.exists()
        assert self.tokenizer.model_max_length > self.args.num_key_frame
        basic_info_path = self.args.output_dir / "basic_info.json"
        tokenized_data_file_path = self.args.output_dir / "tokenized_data.pkl"

        
        with self.args.caption_file_path.open(mode="r") as f:
            coco_meta_data = json.load(f)

            image_id2feature_file_path = {}
            for data in coco_meta_data["images"]:
                feature_file_name = f"{Path(data['file_name']).stem}_{self.args.feature_model_name}.npy"
                feature_path = self.args.feature_dir / feature_file_name
                assert feature_path.exists()
                image_id2feature_file_path[data["id"]] = feature_path

            image_id2caption = {}
            for data in coco_meta_data["annotations"]:
                image_id2caption[data["image_id"]] = data["caption"]

        del coco_meta_data
        
        image_ids = list(image_id2feature_file_path.keys())
        count = 0
        pbar = tqdm(total=self.args.num_instances)
        with PickleFileWriter(tokenized_data_file_path) as pickle_writer:
            while count < self.args.num_instances:
                image_ids = random.sample(image_ids, k=len(image_ids))
                
                for chunk in chunked(image_ids, self.args.num_key_frame):
                    if len(chunk) != self.args.num_key_frame:
                        break

                    chunk_captions = [
                        image_id2caption[image_id]
                        for image_id in chunk
                    ]
                    chunk_feature_paths = [
                        image_id2feature_file_path[image_id]
                        for image_id in chunk
                    ]

                    label_tensors = []
                    frame_locations = []
                    
                    if self.args.post_frame:
                        next_frame_location = -2
                    else:
                        next_frame_location = 0
                    
                    with self.tokenizer.as_target_tokenizer():
                        for caption in chunk_captions:
                            tokenized_caption_ids = self.tokenizer(
                                caption.strip(),
                                padding=False,
                                truncation=True,
                                return_tensors='pt'
                            ).input_ids[0][:-1]
                            
                            if self.args.post_frame:
                                label_tensors.append(tokenized_caption_ids)
                                label_tensors.append(
                                    torch.tensor(
                                        self.tokenizer.pad_token_id,
                                        dtype=torch.long,
                                    ).view(-1)
                                )
                                label_tensors.append(
                                    torch.tensor(
                                        self.bos_token_id,
                                        dtype=tokenized_caption_ids.dtype,
                                    ).view(-1)
                                )
                                next_frame_location += (len(tokenized_caption_ids) + 1 + 1)
                                frame_locations.append(next_frame_location)
                            else:
                                label_tensors.append(
                                    torch.tensor(
                                        self.tokenizer.pad_token_id,
                                        dtype=torch.long,
                                    ).view(-1)
                                )
                                frame_locations.append(next_frame_location)
                                label_tensors.append(tokenized_caption_ids)
                                label_tensors.append(
                                    torch.tensor(
                                        self.bos_token_id,
                                        dtype=tokenized_caption_ids.dtype,
                                    ).view(-1)
                                )
                                # frame_id + len(tokenized_caption_ids) + bos_token_id
                                next_frame_location += (1 + len(tokenized_caption_ids) + 1)

                            
                    # Add eos token
                    label_tensors = label_tensors[:-1]
                    label_tensors.append(
                        torch.tensor(
                            self.tokenizer.eos_token_id,
                            dtype=tokenized_caption_ids.dtype,
                        ).view(-1)
                    )

                    # Truncate frame informations
                    frame_locations = list(
                        filter(
                            lambda i: i < self.tokenizer.model_max_length,
                            frame_locations,
                        )
                    )
                    frame_locations = torch.tensor(frame_locations, dtype=torch.long)
                    
                    # Concat and truncation
                    labels = torch.cat(
                        label_tensors, dim=0
                    )[:self.tokenizer.model_max_length]
                    assert torch.all(
                        torch.index_select(
                            labels, 0, frame_locations, 
                        ) == self.tokenizer.pad_token_id,
                        dim=0,
                    )

                    instance = {
                        "labels": labels,
                        "frame_locations": frame_locations,
                        "feature_paths": chunk_feature_paths,
                        "image_ids": chunk,
                    }
                    pickle_writer.write(instance)
                    count += 1
                    pbar.update()
                    if count == self.args.num_instances:
                        break
        
        basic_info = {
            "collator": "RerankPsedoVideoDataCollator",
            "data_size": count,
            "caption_file_path": str(self.args.caption_file_path),
            "feature_dir": str(self.args.feature_dir),
        }
        with basic_info_path.open(mode="w") as f:
            json.dump(basic_info, f, indent=4)
        logger.info(f"Conclusion: {json.dumps(basic_info, indent=4)}")
        



class PicklizeRerankVISTPseudoVideoPreProcessor(PicklizeRerankCocoPseudoVideoPreProcessor):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--meta_data_file_path",
            help="Specify caption_file_path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--output_dir",
            help="Specify output_dir",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--feature_dir",
            help="Specify output_json_path",
            type=Path,
            required=True
        )
        parser.add_argument(
            "--feature_model_name",
            help="Specify model name to create feature",
            type=str,
            default="clip",
        )
        parser.add_argument(
            "--tokenizer_name_or_path",
            help="Specify tokenizer name or path",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--subset_size",
            help="Specify subset size",
            type=int,
        )
        parser.add_argument(
            "--seed",
            help="Specify seed",
            type=int,
            default="42",
        )
        parser.add_argument(
            "--num_key_frame",
            help="Specify minimum number of key frame. if not specified, all anotated frame will be used.",
            type=int,
        )
        parser.add_argument(
            "--model_max_length",
            help="Specify number of max input tokens in encoder.",
            type=int,
        )
        parser.add_argument(
            "--post_frame",
            help="Whether to use post frame mode",
            action="store_true",
        )
        parser.add_argument(
            "--parallel",
            action='store_true'
        )
    
    @torch.no_grad()
    def __call__(self):
        assert self.args.feature_dir.exists()
        assert self.tokenizer.model_max_length > self.args.num_key_frame
        basic_info_path = self.args.output_dir / "basic_info.json"
        tokenized_data_file_path = self.args.output_dir / "tokenized_data.pkl"
        
        with self.args.meta_data_file_path.open(mode="r") as f:
            vist_meta_data = json.load(f)
        
        count = 0
        with PickleFileWriter(tokenized_data_file_path) as pickle_writer:
            for story_id, stories in tqdm(
                    vist_meta_data.items(),
                    total=len(vist_meta_data)
            ):
                if len(stories) < self.args.num_key_frame:
                    continue
                stories = stories[:self.args.num_key_frame]
                
                label_tensors = []
                frame_locations = []                
                if self.args.post_frame:
                    next_frame_location = -2
                else:
                    next_frame_location = 0
                    
                with self.tokenizer.as_target_tokenizer():
                    for caption in map(lambda d: d["text"], stories):
                        tokenized_caption_ids = self.tokenizer(
                            caption.strip(),
                            padding=False,
                            truncation=True,
                            return_tensors='pt'
                        ).input_ids[0][:-1]

                        if self.args.post_frame:
                            label_tensors.append(tokenized_caption_ids)
                            label_tensors.append(
                                torch.tensor(
                                    self.tokenizer.pad_token_id,
                                    dtype=torch.long,
                                ).view(-1)
                            )
                            label_tensors.append(
                                torch.tensor(
                                    self.bos_token_id,
                                    dtype=tokenized_caption_ids.dtype,
                                ).view(-1)
                            )
                            next_frame_location += (len(tokenized_caption_ids) + 1 + 1)
                            frame_locations.append(next_frame_location)
                        else:
                            label_tensors.append(
                                torch.tensor(
                                    self.tokenizer.pad_token_id,
                                    dtype=torch.long,
                                ).view(-1)
                            )
                            frame_locations.append(next_frame_location)
                            label_tensors.append(tokenized_caption_ids)
                            label_tensors.append(
                                torch.tensor(
                                    self.bos_token_id,
                                    dtype=tokenized_caption_ids.dtype,
                                ).view(-1)
                            )
                            # frame_id + len(tokenized_caption_ids) + bos_token_id
                            next_frame_location += (1 + len(tokenized_caption_ids) + 1)
            
                # Add eos token
                label_tensors = label_tensors[:-1]
                label_tensors.append(
                    torch.tensor(
                        self.tokenizer.eos_token_id,
                        dtype=tokenized_caption_ids.dtype,
                    ).view(-1)
                )

                # Truncate frame informations
                frame_locations = list(
                    filter(
                        lambda i: i < self.tokenizer.model_max_length,
                        frame_locations,
                    )
                )
                frame_locations = torch.tensor(frame_locations, dtype=torch.long)

                # Concat and truncation
                labels = torch.cat(
                    label_tensors, dim=0
                )[:self.tokenizer.model_max_length]
                assert torch.all(
                    torch.index_select(
                        labels, 0, frame_locations, 
                    ) == self.tokenizer.pad_token_id,
                    dim=0,
                )
                
                instance = {
                    "labels": labels,
                    "frame_locations": frame_locations,
                    "feature_paths": [Path(d["feature_path"]) for d in stories],
                    # "image_paths": [Path(d["image_path"]) for d in stories],
                }
                pickle_writer.write(instance)
                count += 1
                
        basic_info = {
            "collator": "RerankPsedoVideoDataCollator",
            "data_size": count,
            "caption_file_path": str(self.args.meta_data_file_path),
            "feature_dir": str(self.args.feature_dir),
        }
        with basic_info_path.open(mode="w") as f:
            json.dump(basic_info, f, indent=4)
        logger.info(f"Conclusion: {json.dumps(basic_info, indent=4)}")
