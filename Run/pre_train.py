import os
import torch
from math import sqrt

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from GlobWeather.data.multi_step_datamodule import MultiStepDataRandomizedModule
from GlobWeather.models.iterative_module import GlobalForecastIterativeModule
from GlobWeather.models.hub.Arrow import Block, WeatherEmbedding

from swanlab.integration.pytorch_lightning import SwanLabLogger

torch.set_float32_matmul_precision('high')

class CustomCLI(LightningCLI):
    def setup_parser(self, *args, **kwargs):
        import datetime
        from omegaconf import OmegaConf
        
        def timestamp_fn(fmt=""):
            return datetime.datetime.now().strftime(fmt or "%m%d_%H%M")
        
        OmegaConf.register_new_resolver("timestamp", timestamp_fn)
        super().setup_parser(*args, **kwargs)

    def before_instantiate_classes(self) -> None:
        pass

    def _instantiate_trainer(self, config, callbacks):
        key = "callbacks"
        if key in config:
            if config[key] is None:
                config[key] = []
            elif not isinstance(config[key], list):
                config[key] = [config[key]]
            config[key].extend(callbacks)
            if key in self.trainer_defaults:
                value = self.trainer_defaults[key]
                config[key] += value if isinstance(value, list) else [value]
            if self.save_config_callback and not config.get("fast_dev_run", False):
                config_callback = self.save_config_callback(
                    self._parser(self.subcommand),
                    self.config.get(str(self.subcommand), self.config),
                    **self.save_config_kwargs,
                )
                config[key].append(config_callback)
        else:
            rank_zero_warn(
                f"The `{self.trainer_class.__qualname__}` class does not expose the `{key}` argument so they will"
                " not be included."
            )
        
        if config['strategy'] == 'fsdp':
            fsdp_strategy = FSDPStrategy(
                sharding_strategy="SHARD_GRAD_OP",
                activation_checkpointing_policy={Block, WeatherEmbedding},
                auto_wrap_policy={Block, WeatherEmbedding}
            )
            config['strategy'] = fsdp_strategy
        elif config['strategy'] == 'ddp':
            config['strategy'] = DDPStrategy(find_unused_parameters=True)
        
        return self.trainer_class(**config)
    
    def adaptive_lr(self):
        basic_lr = self.config.model.lr
        basic_warmup_lr = self.config.model.warmup_start_lr
        basic_eta_min = self.config.model.eta_min
        bs = self.config.data.batch_size
        acc_grad_bs = self.config.trainer.accumulate_grad_batches

        if isinstance(self.config.trainer.devices, str):
            ngpu = len(self.config.trainer.devices.split(','))
        else:
            ngpu = self.config.trainer.devices

        self.config.model.lr = acc_grad_bs * ngpu * (bs / 512) * basic_lr
        self.config.model.warmup_start_lr = acc_grad_bs * ngpu * (bs / 512) * basic_warmup_lr
        self.config.model.eta_min = acc_grad_bs * ngpu * (bs / 512) * basic_eta_min 
        print(f"lr: {self.config.model.lr}, warmup_start_lr: {self.config.model.warmup_start_lr}, eta_min: {self.config.model.eta_min}")

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = CustomCLI(
        model_class=GlobalForecastIterativeModule,
        datamodule_class=MultiStepDataRandomizedModule,
        seed_everything_default=42,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_transforms(*cli.datamodule.get_transforms())
    cli.model.set_clim()
    
    cli.model.set_base_intervals_and_lead_times(
        cli.datamodule.hparams.list_train_intervals,
        cli.datamodule.hparams.val_lead_times
    )

    for i in range(len(cli.trainer.callbacks)):
        if isinstance(cli.trainer.callbacks[i], ModelCheckpoint):
            cli.trainer.callbacks[i] = ModelCheckpoint(
                dirpath=os.path.join(cli.trainer.default_root_dir, "checkpoints"),
                monitor=cli.trainer.callbacks[i].monitor,
                mode=cli.trainer.callbacks[i].mode,
                save_top_k=cli.trainer.callbacks[i].save_top_k,
                save_last=cli.trainer.callbacks[i].save_last,
                verbose=cli.trainer.callbacks[i].verbose,
                filename=cli.trainer.callbacks[i].filename,
                auto_insert_metric_name=cli.trainer.callbacks[i].auto_insert_metric_name
            )

    cli.trainer.logger = SwanLabLogger(
        experiment_name=cli.trainer.logger._experiment_name,
        project=cli.trainer.logger._swanlab_init['project'],
        logdir=os.path.join(cli.trainer.default_root_dir, cli.trainer.logger.save_dir),
        save_dir=os.path.join(cli.trainer.default_root_dir, cli.trainer.logger.save_dir),
        mode=cli.trainer.logger._swanlab_init['mode']
    )

    if os.path.exists(os.path.join(cli.trainer.default_root_dir, 'checkpoints', 'last.ckpt')):
        ckpt_resume_path = os.path.join(cli.trainer.default_root_dir, 'checkpoints', 'last.ckpt')
    else:
        ckpt_resume_path = None

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_resume_path)


if __name__ == "__main__":
    main()