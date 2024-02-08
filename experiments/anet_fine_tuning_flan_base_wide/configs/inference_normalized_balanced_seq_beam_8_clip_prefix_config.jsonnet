local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local LEARNING_TYPE = "no_meta_learning";
local SHARE_DIR = std.extVar("SHARE_DIR");
local utils = import "./utils.jsonnet";

local train_config = import "./train_config.jsonnet";
local train_data_config = import "./prepro_config_train.jsonnet";
local inference_data_config = import "./prepro_config_clip_prefix.jsonnet";

train_config + {
  global_setting: super.global_setting + {
    pl_model_name: "Video2TextRerankerT5InferencePL",
    load_check_point: "best",
  },
  
  Logger: super.Logger + {
    version: "%s_clip_prefix_seq_beam_8_prioritize_balanced_normalized/inference" % $.global_setting.tag,
  },
  
  pl_module_setting: super.pl_module_setting + {
    num_key_frame: train_data_config.num_key_frame,
    caption_json_file_path: inference_data_config.caption_json_file_path,
    seq_beam: 8,
    normalized_score: true,
    ranking_score_epsilon: 0.5,
  },
  
  Datasets: super.Datasets + {
    test_data_file_paths: [
      inference_data_config.output_dir,
    ],
    batch_size: 32,
    num_workers: 1,
  },
  
  train_only: false,
}
