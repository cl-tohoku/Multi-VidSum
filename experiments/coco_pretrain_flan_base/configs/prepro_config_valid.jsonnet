local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DOWNLOAD_DIR = "%s/download" % ROOT_DIR;
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local prepro_train_config = import "./prepro_config_train.jsonnet";

prepro_train_config + {
  seed: super.seed + 1,
  caption_file_path: "%s/coco/annotations/captions_train2014.json" % DOWNLOAD_DIR,
  output_dir: "%s/%s_valid_num_key_frame_%s" % [
    DATASET_DIR,
    std.extVar("TAG"),
    std.toString(self.num_key_frame),
  ],
  num_instances: 1000,
}
