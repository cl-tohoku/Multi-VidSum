local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local DOWNLOAD_DIR = "%s/download" % ROOT_DIR;

local train_data_config = import "./prepro_config_train.jsonnet";
local valid_data_config = import "./prepro_config_valid.jsonnet";
local test_data_config = import "./prepro_config_test.jsonnet";

{
  train_only: true,
  
  global_setting: {
    pl_model_name: "Video2TextRerankerT5PL",
    seed: 42,
    tag: "%s_seed_%s" % [
      std.extVar("TAG"),
      std.toString(self.seed),
    ],
    log_model_output_dir: "%s/experiment_results/%s" % [
      ROOT_DIR,
      self.tag,
    ],
  },
  
  Logger: {
    project_name: "multi_vid_sum",
    log_dir: "%s/logs" % $.global_setting.log_model_output_dir,
    version: "%s/train" % [$.global_setting.tag],
  },
  
  Trainer: {
    max_epochs: 200,
    val_check_interval: "1.0",
    check_val_every_n_epoch: 1,
    default_root_dir: "%s/defaults" % $.global_setting.log_model_output_dir,
    weights_save_path: "%s/weights" % $.global_setting.log_model_output_dir,
    strategy : "ddp",
    bf16: true,
    accumulate_grad_batches: 8,
  },
  
  Callbacks: {
    save_top_k: 2,
    checkpoint_save_path: "%s/checkpoints" % $.global_setting.log_model_output_dir,
    early_stopping_patience: -1,
    async_checkpointing: true,
  },
  
  pl_module_setting: {
    lr_scheduler: "cosine_scheduler",
    lr: 1e-3,
    num_warmup_steps: 500,
    
    model_name_or_path: "%s/experiment_results/vist_pretrain_flan_base_seed_%s/weights/Video2TextRerankerPseudoVideoPretrainT5PL" % [
      ROOT_DIR,
      std.toString($.global_setting.seed),
    ],
    
    # When using pretrained model, specify folloeing model_name_or_path
    # model_name_or_path: "%s/pretrained_models/Video2TextRerankerT5PL" % DOWNLOAD_DIR,
    
    model_max_length: train_data_config.model_max_length,
    from_scratch: false,
    tokenizer_name_or_path: train_data_config.tokenizer_name_or_path,
    
    num_beams: 1,
    max_length: self.model_max_length,
    
    length_penalty: 0.0,
    num_wandb_upload: 30,
    
    fix_embedding: false,
    # use_frame_loss: true,
    is_pointer_network: true,
    with_discrete_gate: true,
    normalize_last_hidden_states: true,
    separated_cross_entropy_loss: true,
    loss_frame_weight: 0.5,
  },
  
  Datasets: {
    train_data_file_path: train_data_config.output_dir,
    valid_data_file_path: valid_data_config.output_dir,
    test_data_file_paths: [
      test_data_config.output_dir,
    ],
    batch_size: 6,
    num_workers: 32,
    train_data_shuffle: true,
  },
}
