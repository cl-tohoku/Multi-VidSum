local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;
local DOWNLOAD_DIR = "%s/download" % ROOT_DIR;
local utils = import "./utils.jsonnet";

local prepro_valid_config = import "./prepro_config_valid.jsonnet";


utils.objectPop(prepro_valid_config, "split") + {
  output_dir: "%s/%s_%s_rerank_%s_key_frame_%s_sampling_%s_frep" % [
    DATASET_DIR,
    std.extVar("TAG"),
    "test",
    std.toString(self.num_key_frame),
    std.toString(self.num_sampling),
    std.format("%.3f", self.discretized_frequency),
  ],
  meta_data_path: "%s/anet/annotations/eval_meta_data.jsonl" % [
    DOWNLOAD_DIR,
  ],
  feature_dir: "%s/anet/features/eval" % [
    DOWNLOAD_DIR,
  ],
  num_sampling: 1,
  use_insufficient_key_frame_movies: true,
}
