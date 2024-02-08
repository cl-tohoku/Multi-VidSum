local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DOWNLOAD_DIR = "%s/download" % ROOT_DIR;
local DATASET_DIR = "%s/datasets" % ROOT_DIR;

local prepro_valid_config = import "./prepro_config_valid.jsonnet";

prepro_valid_config + {
  caption_file_path: "%s/coco/annotations/captions_val2014.json" % DOWNLOAD_DIR,
  output_dir: "%s/%s_test_num_key_frame_%s" % [
    DATASET_DIR,
    std.extVar("TAG"),
    std.toString(self.num_key_frame),
  ],
  feature_dir: "%s/coco/val2014_features" % [
    DOWNLOAD_DIR,
  ],
  num_instances: 1000,
}
