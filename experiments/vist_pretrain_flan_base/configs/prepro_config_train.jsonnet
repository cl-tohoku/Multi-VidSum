local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;
local DOWNLOAD_DIR = "%s/download" % ROOT_DIR;

local split = "train";

{
  preprocessor_name: "PicklizeRerankVISTPseudoVideoPreProcessor",
  seed: 42,
  meta_data_file_path: "%s/vist/meta_data/%s.json" % [
    DOWNLOAD_DIR,
    split,
  ],
  output_dir: "%s/%s_%s_num_key_frame_%s" % [
    DATASET_DIR,
    split,
    std.extVar("TAG"),
    std.toString(self.num_key_frame),
  ],
  feature_dir: "%s/vist/features/%s" % [
    DOWNLOAD_DIR,
    split,
  ],
  tokenizer_name_or_path: "google/flan-t5-base",
  num_key_frame: 4,
  model_max_length: 2048,
}
