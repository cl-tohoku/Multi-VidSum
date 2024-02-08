local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;

local train_config = import "./train_config.jsonnet";

train_config + {
  Logger: train_config.Logger + {
    version: "%s/test_beam_%s" % [
      $.global_setting.tag,
      std.toString($.pl_module_setting.num_beams),
    ],
  },

  global_setting: train_config.global_setting + {
    load_check_point: "best",
  },

  pl_module_setting: super.pl_module_setting + {
    num_beams: 16,
  },
  
  Datasets: train_config.Datasets + {
    batch_size: 4,
  },
  
  train_only: false,
}
