local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local prepro_train_config = import "./prepro_config_train.jsonnet";

prepro_train_config + {
  seed: super.seed + 1,
  output_dir: "%s/%s_%s_rerank_%s_key_frame_%s_sampling_%s_frep" % [
    DATASET_DIR,
    std.extVar("TAG"),
    "valid",
    std.toString(self.num_key_frame),
    std.toString(self.num_sampling),
    std.format("%.3f", self.discretized_frequency),
  ],
  num_sampling: 1,
}
