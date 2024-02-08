local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATASET_DIR = "%s/datasets" % ROOT_DIR;
local DOWNLOAD_DIR = "%s/download" % ROOT_DIR;

{
  preprocessor_name: "PicklizeRerankCocoPseudoVideoPreProcessor",
  seed: 42,
  caption_file_path: "%s/coco/annotations/captions_train2014.json" % DOWNLOAD_DIR,
  output_dir: "%s/%s_train_num_key_frame_%s" % [
    DATASET_DIR,
    std.extVar("TAG"),
    std.toString(self.num_key_frame),
  ],
  feature_dir: "%s/coco/train2014_features" % [
    DOWNLOAD_DIR,
  ],
  num_instances: 100000,
  tokenizer_name_or_path: "google/flan-t5-base",
  num_key_frame: 4,
  model_max_length: 2048,
  # subset_size: 10,
}
