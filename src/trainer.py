import argparse
from pathlib import Path
import os
from datetime import datetime
import json

from logzero import logger
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins.io import AsyncCheckpointIO
#from pytorch_lightning.strategies.ddp import DDPStrategy

import models
from pickle_dataset import PickleDataset
from utils.argparse_helpers import float_or_int

class Trainer:
    def __init__(self, args, mode="train"):
        self.args = args
        self.mode = mode.lower()
        self.current_check_point = None
        seed_everything(self.args.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        assert os.environ.get("PYTHONHASHSEED") == "0",\
            "Set enviroment variable \"PYTHONHASHSEED\" = \"0\""
        os.environ["TOKENIZERS_PARALLELISM"] = str(args.num_workers == 0).lower()
        
        cls = getattr(models, self.args.pl_model_name)
        logger.info(f"Selected model : {args.pl_model_name}")

        
        if self.args.load_check_point is not None:
            if str(self.args.load_check_point) == "best":
                with (self.args.default_root_dir / "best_model_path.text").open(mode="r") as f:
                    self.args.load_check_point = f.read().strip()
            
            logger.info(f"Load model from \"{args.load_check_point}\"")
            self.pl_model = cls.load_from_checkpoint(
                self.args.load_check_point,
                config=args,
            )
            self.current_check_point = Path(self.args.load_check_point)
        else:
            logger.info("Learning from beggning.")
            self.pl_model = cls(self.args)
        
        self.load_datasets()

        self.args.log_model_output_dir.mkdir(parents=True, exist_ok=True)
        self.args.default_root_dir.mkdir(parents=True, exist_ok=True)
        self.args.weights_save_path.mkdir(parents=True, exist_ok=True)
        self.args.log_dir.mkdir(parents=True, exist_ok=True)
        self.args.checkpoint_save_path.mkdir(parents=True, exist_ok=True)

        self.pl_trainer_setting()
        

    def load_datasets(self):
        if self.mode == "train":
            
            self.train_dataset = PickleDataset(self.args.train_data_file_path)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                collate_fn=self.train_dataset.collator(self.args, self.pl_model.tokenizer),
                shuffle=self.args.train_data_shuffle,
            )

            self.valid_dataset = PickleDataset(self.args.valid_data_file_path)
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                collate_fn=self.valid_dataset.collator(self.args, self.pl_model.tokenizer),
                shuffle=False,
            )

        self.test_datasets = []
        self.test_dataloaders = []
        for path in self.args.test_data_file_paths:
            self.test_datasets.append(PickleDataset(path))
            self.test_dataloaders.append(
                DataLoader(
                    self.test_datasets[-1],
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    collate_fn=self.test_datasets[-1].collator(self.args, self.pl_model.tokenizer),
                    shuffle=False,
                )
            )

    


    def pl_trainer_setting(self):
        start_time = datetime.today().strftime("%Y%m%d%H%M%S")

        # save_dirは必ずstrにする
        pl_logger = WandbLogger(
            save_dir=str(self.args.log_dir),
            name=f"{self.args.version}_{start_time}",
            project=self.args.project_name
        )
        
        check_point_filename = "checkpoint-{epoch:02d}-{step:02d}-{valid_loss:.4f}-" + start_time
        
        callbacks = [
            ModelCheckpoint(
                monitor=self.args.monitor,
                mode=self.args.stop_mode,
                verbose=True,
                dirpath=self.args.checkpoint_save_path,
                save_top_k=self.args.save_top_k,
                filename=check_point_filename,
                save_last=True,
            ),
            
            LearningRateMonitor(
                logging_interval="step",
            ),
        ]
        
        
        if self.args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor=self.args.monitor,
                    min_delta=0.0,
                    patience=self.args.early_stopping_patience,
                    mode=self.args.stop_mode,
                    stopping_threshold=self.args.stopping_threshold,
                    check_on_train_epoch_end=not self.args.check_on_each_evaluation_step,
                    strict=True,
                )
            )
        else:
            logger.info("Don't use EarlyStopping!")
        

        if self.args.max_epochs == -1:
            self.args.max_epochs = None
        
        # Specify fp16
        if self.args.fp16:
            logger.info("Use fp16!")
            precision = 16
        elif self.args.bf16:
            precision = "bf16"
        else:
            precision = 32


        # Specify apex
        if self.args.apex:
            logger.info("Use apex!")
            amp_backend = "apex"
        else:
            amp_backend = "native"
            assert self.args.amp_level is None

        

            
        # Specify gpu id
        # if self.mode == "test":
        #     assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) <= 1, \
        #         "You can use only 1 gpu for test mode"
        logger.info(
            f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
        assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == torch.cuda.device_count(), "Number of visible GPU error"            
        

        
        if torch.cuda.device_count() != 0:
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = [-1]
        logger.info(f"GPU IDS: {gpu_ids}")
        
        
        # Calc frequency of writting log to disk
        log_every_n_steps =  self.args.log_every_n_steps
        
        assert (not self.args.accelerator == "cpu") or gpu_ids == [-1], \
            "Specify accelerator to cpu!"
        assert (self.args.max_steps != -1) or (self.args.max_epochs != -1)


        plugins = []
        if self.args.async_checkpointing:
            plugins.append(AsyncCheckpointIO())

        self.pl_trainer = pl.Trainer(
            check_val_every_n_epoch=self.args.check_val_every_n_epoch,
            val_check_interval=self.args.val_check_interval,
            deterministic=True,
            callbacks=callbacks,
            devices=gpu_ids,
            accelerator=self.args.accelerator,
            strategy=self.args.strategy,
            fast_dev_run=self.args.fast_dev_run,
            log_every_n_steps=log_every_n_steps,
            max_epochs=self.args.max_epochs,
            max_steps=self.args.max_steps,
            profiler=None,
            reload_dataloaders_every_n_epochs=0,
            default_root_dir=self.args.default_root_dir,
            logger=pl_logger,
            precision=precision,
            gradient_clip_val=self.args.gradient_clip_val,
            accumulate_grad_batches=self.args.accumulate_grad_batches,
            amp_backend=amp_backend,
            amp_level=self.args.amp_level,
            plugins=plugins,
        )
        
    
    def test(self):
        logger.info("Test start!!")
        test_result_dict = {
            "tag": self.args.tag,
            "results": []
        }
        
        for idx, (path, dataloader) in enumerate(
                zip(
                    self.args.test_data_file_paths,
                    self.test_dataloaders
                )
        ):
            self.pl_model.test_data_idx = idx
            test_result = self.pl_trainer.test(
                self.pl_model,
                dataloaders=dataloader,
                ckpt_path=self.args.load_check_point,
            )
            assert len(test_result) == 1, "Expect to test one dataset for each loop."
            test_result = test_result[0]
            test_result["idx"] = idx
            test_result["dataset_path"] = str(path)
            test_result["check_point_path"] = str(self.current_check_point)
            test_result_dict["results"].append(test_result)
            
        now = datetime.today().strftime("%Y%m%d%H%M%S")
        test_result_file_name = f"test_result_{now}.json"
        with (self.args.default_root_dir / test_result_file_name).open(mode="w") as f:
            json.dump(test_result_dict, f)
        
        logger.info("Test finish!!")

        return test_result_dict
    
    def train(self):
        assert self.mode == "train", "Set seelf.mode = \"train\"!"
        
        # Start train!
        logger.info("Train start!!")
        self.pl_trainer.fit(
            self.pl_model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader,
            ckpt_path=self.args.load_check_point,
        )
        logger.info("Train finish!!")
        
        if self.pl_trainer.is_global_zero:
            with (self.args.default_root_dir / "best_model_path.text").open(mode="w") as f:
                best_model_path = str(self.pl_trainer.checkpoint_callback.best_model_path)
                if best_model_path == "":
                    logger.warning("No best_model_path exists...")
                else:  
                    f.write(best_model_path)
                    logger.info("Save transformers model")
            
            model_weight_save_path = self.args.weights_save_path / self.args.pl_model_name
            model_weight_save_path.mkdir(parents=True, exist_ok=True)
            self.pl_model.model.save_pretrained(model_weight_save_path)
    

    def __call__(self, train_only=False):
        # train
        self.train()
        
        if not train_only:
            if self.args.strategy is not None:
                torch.distributed.destroy_process_group()
                if self.trainer.is_global_zero:
                    self.pl_trainer.training_type_plugin.num_nodes = 1
                    self.pl_trainer.training_type_plugin.num_processes = 1
                    self.pl_trainer.accelerator_connector.replace_sampler_ddp = False
                else:
                    return
            
            try:
                self.pl_model = self.pl_model.__class__.load_from_checkpoint(
                    self.pl_trainer.checkpoint_callback.best_model_path,
                    strict=False,
                )
            except Exception as e:
                print(f"({e})")
                logger.error("No checkpoint were saved...")
                return 
            
            self.current_check_point = Path(self.pl_trainer.checkpoint_callback.best_model_path)
            logger.info(f"Load best model from {self.pl_trainer.checkpoint_callback.best_model_path}")
            return self.test()
        
        else:
            return

    
    
    @staticmethod
    def add_args(parent_parser):
        initial_parser = argparse.ArgumentParser()
        initial_parser.add_argument("--pl_model_name", help="Select pl_model_name", type=str, required=True)
        initial_args, unrecognized = initial_parser.parse_known_args()
        cls = getattr(models, initial_args.pl_model_name)
        cls.add_model_specific_args(parent_parser)
        
        parent_parser = Trainer.add_global_setting_args(parent_parser)
        parent_parser = Trainer.add_trainer_setting_args(parent_parser)
        parent_parser = Trainer.add_logger_setting_args(parent_parser)
        parent_parser = Trainer.add_callbacks_args(parent_parser)
        parent_parser = Trainer.add_dataset_args(parent_parser)
        return parent_parser


    
    @staticmethod
    def add_global_setting_args(parent_parser):
        parser = parent_parser.add_argument_group("global_setting")
        parser.add_argument("--pl_model_name", help="Specify pl_model_name", type=str, required=True)
        parser.add_argument("--seed", help="Specify random seed", type=int, required=True)
        #parser.add_argument("--gpu_id", help="Specify gpu id", nargs='*', type=int, default=None)
        parser.add_argument("--tag", help="Specify tag.", type=str, required=True)
        parser.add_argument("--log_model_output_dir", help="Specify log_model_output_dir.", type=Path, required=True)
        parser.add_argument("--load_check_point", help="Specify checkpoint", type=Path)
        return parent_parser


    @staticmethod
    def add_trainer_setting_args(parent_parser):
        parser = parent_parser.add_argument_group("Trainer")
        parser.add_argument("--max_epochs", type=int, default=-1)
        parser.add_argument("--max_steps", type=int, default=-1)    
        parser.add_argument("--check_val_every_n_epoch", help="Specify frequency of validation steps.", type=int)
        parser.add_argument("--val_check_interval", help="Specify frequency of validation steps.", type=float_or_int, required=True)
        parser.add_argument("--default_root_dir", help="Specify default root dir.", type=Path, required=True)
        parser.add_argument("--weights_save_path", help="Specify weights save path.", type=Path, required=True)
        parser.add_argument("--fp16", help="Specify whether to use fp16", action="store_true")
        parser.add_argument("--bf16", help="Specify whether to use fp16", action="store_true")
        parser.add_argument("--apex", help="Specify whether to use apex", action="store_true")
        parser.add_argument("--amp_level", help="Specify whether to use apex", type=str)
        parser.add_argument("--fast_dev_run", help="Specify fast_dev_run step", type=int, default=0)
        parser.add_argument("--accelerator", help="Specify accelerator", type=str, default="gpu")
        parser.add_argument("--strategy", help="Specify strategy", type=str)
        parser.add_argument("--gradient_clip_val", help="Specify gradient_clip_val", type=float)
        parser.add_argument("--accumulate_grad_batches", help="Specify accumulate_grad_batches", type=int)
        return parent_parser

    @staticmethod
    def add_logger_setting_args(parent_parser):
        parser = parent_parser.add_argument_group("Logger")
        parser.add_argument("--project_name", help="Specify wandb project name.", type=str, required=True)
        parser.add_argument("--log_dir", help="Specify log dir.", type=Path, required=True)
        parser.add_argument("--version", help="Specify log version.", type=str, default="")
        parser.add_argument("--log_every_n_steps", help="Specify log_every_n_steps.", type=int, default=10)
        return parent_parser

    @staticmethod
    def add_callbacks_args(parent_parser):
        parser = parent_parser.add_argument_group("Callbacks")
        parser.add_argument("--checkpoint_save_path", help="Specify checkpoint_save_path.", type=Path, required=True)
        parser.add_argument("--save_top_k", help="Specify checkpoint_save_path.", type=int, required=True)
        parser.add_argument("--early_stopping_patience", help="Specify early_stopping_patience.", type=int, required=True)
        parser.add_argument("--monitor", help="Specify monitor.", type=str, default="valid_loss")
        parser.add_argument("--stop_mode", help="Specify stop mode.", type=str, default="min")
        parser.add_argument("--stopping_threshold", help="Specify stopping_threshold.", type=float)
        parser.add_argument("--check_on_each_evaluation_step", help="Specify check_on_each_evaluation_step mode", action="store_true")
        parser.add_argument("--async_checkpointing", help="Specify async checkpointing mode", action="store_true")
        return parent_parser
    

    @staticmethod
    def add_dataset_args(parent_parser):
        parser = parent_parser.add_argument_group("Datasets")
        parser.add_argument("--train_data_file_path", help="Specify train data file path", type=Path, required=True)
        parser.add_argument("--valid_data_file_path", help="Specify valid data file path", type=Path, required=True)
        parser.add_argument("--test_data_file_paths", help="Specify test data file paths", nargs='*', type=Path, default=[])
        parser.add_argument("--batch_size", help="Specify batch size.", type=int, required=True)
        parser.add_argument("--num_workers", help="Specify number of workers.", type=int, default=0)
        parser.add_argument("--train_data_shuffle", help="Specify whether shuffle train data or not", action="store_true")
        return parent_parser
    







def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
