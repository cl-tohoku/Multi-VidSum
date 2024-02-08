import argparse
from pathlib import Path
import random

from logzero import logger
import torch
from torch import nn
from transformers import T5Config, T5TokenizerFast, Video2TextConditionalGeneration, T5ForConditionalGeneration
from tqdm.auto import tqdm



def test(model, args):
    config = model.config
    batch_size = 8
    seq_len = config.n_positions
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i in range(batch_size):
        attention_mask[i, 0:random.randrange(seq_len)] = 1
    
    batch = {
        "inputs_embeds": torch.randn(batch_size, seq_len, config.feature_embed_dim),
        "attention_mask": attention_mask,
        "labels": torch.randint(0, config.vocab_size, (batch_size, 100))
    }
    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        model = model.to(device)
        for k, v in batch.items():
            batch[k] = v.to(device)
            
    output = model(**batch)
    logger.info("OK!")
    model.cpu()
    del batch


def main(args):
    args.save_dir.mkdir(parents=True, exist_ok=True)
    # pretrained_model_names = [
    #     "t5-base", "t5-large", "t5-3b",
    # ]
    # pretrained_model_names = [
    #     "t5-base",
    # ]
    # pretrained_model_names = [
    #     "google/flan-t5-base", "google/flan-t5-large"
    # ]
    pretrained_model_names = [
        "google/flan-t5-base"
    ]

    for name in tqdm(pretrained_model_names):
        tokenizer = T5TokenizerFast.from_pretrained(name)
        config = T5Config.from_pretrained(name)

        if args.n_positions is not None:
            config.n_positions = args.n_positions
            tokenizer.model_max_length = args.n_positions
            
        assert config.n_positions == tokenizer.model_max_length, \
            f"config.n_positions: {config.n_positions} != tokenizer.model_max_length: {tokenizer.model_max_length}"
        
        model = T5ForConditionalGeneration.from_pretrained(name)
        embed = model.get_input_embeddings()

        num_embeddings, embed_dim = embed.weight.size()
        
        config.is_pointer_network = True
        config.with_gate = True
        config.vocab_size = tokenizer.vocab_size
        config.feature_embed_dim = 768
        config.normalize_last_hidden_states = True
        config.label_smoothing = 0.0
        config.separated_cross_entropy_loss = False
        
        if config.is_pointer_network:
            new_embed_weight = embed.weight[:config.vocab_size]
            assert new_embed_weight.size() == (config.vocab_size, embed_dim)
        else:
            frame_embed = torch.randn(
                config.n_positions,
                embed_dim,
                dtype=embed.weight.dtype
            )
            new_embed_weight = torch.cat(
                [
                    embed.weight[:config.vocab_size],
                    frame_embed,
                ],
                dim=0,
            )
            assert new_embed_weight.size() == (config.vocab_size + config.n_positions, embed_dim)
            
        video_model = Video2TextConditionalGeneration.from_pretrained(
            name,
            config=config,
            ignore_mismatched_sizes=True,
        )
        new_embedding = nn.Embedding.from_pretrained(
            new_embed_weight,
            freeze=False,
        )
        video_model.set_input_embeddings(new_embedding)   
        logger.info(f"Finish converting {name} to Video2TextConditionalGeneration")
        test(video_model, args)
        
        save_path = args.save_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        video_model.save_pretrained(save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", help="Specify file path", type=Path, required=True)
    parser.add_argument("--n_positions", help="Specify n_positions", type=int, default=2048)
    parser.add_argument("--gpu_id", help="Specify gpu_id", type=int)
    args = parser.parse_args()
    main(args)

