local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;
local DOWNLOAD_DIR = "%s/download" % ROOT_DIR;

{
  preprocessor_name: "PicklizeRerankPreProcessor",
  seed: 42,
  meta_data_path: "%s/anet/annotations/meta_data.jsonl" % DOWNLOAD_DIR,
  output_dir: "%s/%s_%s_rerank_%s_key_frame_%s_sampling_%s_frep" % [
    DATASET_DIR,
    std.extVar("TAG"),
    self.split,
    std.toString(self.num_key_frame),
    std.toString(self.num_sampling),
    std.format("%.3f", self.discretized_frequency),
  ],
  feature_dir: "%s/anet/features/training" % [
    DOWNLOAD_DIR,
  ],
  tokenizer_name_or_path: "google/flan-t5-base",
  split: "train",
  num_sampling: 8,
  discretized_frequency: 0.5,
  num_key_frame: 4,
  model_max_length: 2048,
  # subset_size: 10,
}
