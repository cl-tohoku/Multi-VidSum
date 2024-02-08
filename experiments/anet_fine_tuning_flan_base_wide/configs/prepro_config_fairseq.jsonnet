local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;
local DOWNLOAD_DIR = "%s/download" % ROOT_DIR;

local prepro_train_config = import "./prepro_config_train.jsonnet";

{
  preprocessor_name: "PicklizeRerankInferencePreProcessor",
  seed: prepro_train_config.seed,
  caption_json_file_path: "%s/all_frame_captions_for_reranking/fairseq_all_frame_captions.json" % DOWNLOAD_DIR,
  output_dir: "%s/%s_fairseq_all_frame_captions" % [
    DATASET_DIR,
    std.extVar("TAG"),
  ],
  feature_dir: "%s/anet/features/eval" % [
    DOWNLOAD_DIR,
  ],
  tokenizer_name_or_path: prepro_train_config.tokenizer_name_or_path,
  model_max_length: prepro_train_config.model_max_length,
  parallel: true,
}
