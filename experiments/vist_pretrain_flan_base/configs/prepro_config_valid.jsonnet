local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;
local DOWNLOAD_DIR = "%s/download" % ROOT_DIR;

local prepro_train_config = import "./prepro_config_train.jsonnet";

local split = "val";

prepro_train_config + {
  seed: super.seed + 1,
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
}

