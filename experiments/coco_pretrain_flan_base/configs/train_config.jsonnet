local ROOT_DIR = std.extVar("ROOT");
local DOWNLOAD_DIR = "%s/download" % ROOT_DIR;

local train_data_config = import "./prepro_config_train.jsonnet";
local valid_data_config = import "./prepro_config_valid.jsonnet";
local test_data_config = import "./prepro_config_test.jsonnet";

{
  train_only: true,
  
  global_setting: {
    pl_model_name: "Video2TextRerankerPseudoVideoPretrainT5PL",
    seed: 42,
    tag: "%s_seed_%s" % [
      std.extVar("TAG"),
      std.toString(self.seed),
    ],
    log_model_output_dir: "%s/experiment_results/%s" % [ROOT_DIR, self.tag],
  },
  
  Logger: {
    project_name: "multi_vid_sum",
    log_dir: "%s/logs" % $.global_setting.log_model_output_dir,
    version: "%s/train" % [$.global_setting.tag],
  },
  
  Trainer: {
    max_epochs: 30,
    val_check_interval: "1.0",
    check_val_every_n_epoch: 1,
    default_root_dir: "%s/defaults" % $.global_setting.log_model_output_dir,
    weights_save_path: "%s/weights" % $.global_setting.log_model_output_dir,
    strategy : "ddp",
    # fp16: true,
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
    lr: 1e-4,
    num_warmup_steps: 1000,
    
    model_name_or_path: "%s/base_models/t5_wide_pointer_network/%s" % [
      DOWNLOAD_DIR,
      train_data_config.tokenizer_name_or_path,
    ],
    model_max_length: train_data_config.model_max_length,
    from_scratch: false,
    tokenizer_name_or_path: train_data_config.tokenizer_name_or_path,
    
    
    num_beams: 1,
    max_length: self.model_max_length,
    
    length_penalty: 0.0,
    num_wandb_upload: 30,
    fix_embedding: false,
    apply_noise: true,
    num_key_frame:  train_data_config.num_key_frame,
    
    is_pointer_network: true,
    with_discrete_gate: true,
    normalize_last_hidden_states: true,
    label_smoothing: 0.1,
    separated_cross_entropy_loss: true,
    loss_frame_weight: 0.5,
  },
  
  Datasets: {
    train_data_file_path: train_data_config.output_dir,
    valid_data_file_path: valid_data_config.output_dir,
    test_data_file_paths: [
      test_data_config.output_dir,
    ],
    batch_size: 3 * 2,
    num_workers: 8,
    train_data_shuffle: true,
  },
}
