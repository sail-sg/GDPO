# These imports are tricky because they use c++, do not move them
from rdkit import Chem
try:
    import graph_tool
except ModuleNotFoundError:
    pass

import os
import pathlib
import warnings
import numpy as np
import random
import torch
import wandb
import hydra
import omegaconf
import sys
from omegaconf import DictConfig
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.loggers import CSVLogger
import utils
from omegaconf import OmegaConf,open_dict
# from dataset.spectre_dataset import IMDBDataModule,SpectreDatasetInfos,PROTEINDataModule,MUTAGDataModule
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
# from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics, ProteinSamplingMetrics, IMDBSamplingMetrics,MUTAGSamplingMetrics
from model.diffusion_discrete import DiscreteDenoisingDiffusion
from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from analysis.visualization import MolecularVisualization, NonMolecularVisualization
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from dataset.ppo_dataset import PPODataModule

warnings.filterwarnings("ignore", category=PossibleUserWarning)
# batch_size = 128

def init(cfg):
    seed = cfg.general.seed
    # seed = random.randint(0,10000)
    device = cfg.general.device
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
import string
 
def generate_random_letter(k):
    letters = string.ascii_letters
    random_letter = "".join(random.choices(letters,k=k))
    return random_letter

def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(
            resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(
            resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    # cfg.train.batch_size = batch_size
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(
            resume_path, cfg=cfg,**model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(
            resume_path, **model_kwargs)
    new_cfg = model.cfg
    for category in cfg:
        for arg in cfg[category]:
            if arg not in new_cfg[category]:
                with open_dict(new_cfg[category]):
                    new_cfg[category][arg] = cfg[category][arg]
            else:
                new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    if cfg.dataset.name in ["zinc","moses"]:
        new_cfg.general.name = new_cfg.general.name + '_resume{}'.format(cfg.general.target_prop)
    else:
        new_cfg.general.name = new_cfg.general.name + '_resume{}'.format(cfg.dataset.name)

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    print("learning rate: {}".format(cfg.train.lr))
    return new_cfg, model


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(version_base='1.1', config_path='./configs', config_name='config')
def main(cfg: DictConfig):
    init(cfg)
    workdir = os.getcwd()
    print("os working dir",workdir)
    if "multirun" in workdir:
        home_prefix = "./../../../../"
    else:
        home_prefix = "./../../../"
    dataset_config = cfg["dataset"]
    print(dataset_config)
    if dataset_config["name"] in ['sbm', 'planar']:
        from dataset.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos, ToyDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)
        if "nodes" in dataset_config:
            dataset_infos = ToyDatasetInfos(datamodule,dataset_config)
            train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
            visualization_tools = NonMolecularVisualization()
        else:
            dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
            train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
            visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    elif dataset_config["name"] in ['moses', "zinc"]:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config.name == "zinc":
            from dataset import zinc_dataset
            datamodule = zinc_dataset.MosesDataModule(cfg)
            dataset_infos = zinc_dataset.MOSESinfos(datamodule,cfg)

            train_smiles = pd.read_csv(home_prefix+"dataset/zinc/raw/zinc_train.csv")["smiles"].tolist()
            dataset_infos.train_smiles = train_smiles
        elif dataset_config.name == 'moses':
            from dataset import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = pd.read_csv(home_prefix+"dataset/moses/moses_pyg/raw/train_moses.csv")["SMILES"].tolist()
            dataset_infos.train_smiles = train_smiles
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = []
    # callbacks.append(lr_monitor)
    if cfg.train.save_model:
        if cfg.general.train_method in ["gdpo","ddpo"]:
            if "nodes" in cfg.dataset:
                topk = 0
            else:
                topk = 50
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                                filename=generate_random_letter(2)+str(cfg.general.seed)+"_"+'{epoch}-{val/epoch_score:.4f}',
                                                monitor="val/epoch_score",
                                                save_top_k=topk,
                                                mode='max',
                                                every_n_train_steps=cfg.general.val_check_interval)
        else:
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                            #   every_n_train_steps=250,
                                            every_n_epochs=1
                                              )
        last_ckpt_save = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)
    print("model loaded, begin training")
    name = cfg.general.name
    if name == 'test':
        print(
            "[WARNING]: Run is called 'test' -- it will run in debug mode on 20 batches. ")
    elif name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    # logger = CSVLogger("logs",name = "graphtest")
    if cfg.general.train_method in ["ddpo","gdpo"]:
        trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                        devices=cfg.general.gpus if torch.cuda.is_available(
                        ) and cfg.general.gpus > 0 else None,
                        limit_train_batches=20 if name == 'test' else None,
                        limit_val_batches=20 if name == 'test' else None,
                        limit_test_batches=20 if name == 'test' else None,
                        val_check_interval=cfg.general.val_check_interval,
                        max_epochs=cfg.train.n_epochs,
                        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                        fast_dev_run=cfg.general.name == 'debug',
                        strategy='ddp' if cfg.general.gpus > 1 else None,
                        enable_progress_bar=False,
                        callbacks=callbacks,
                        logger=[])
    else:
        trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=cfg.general.gpus if torch.cuda.is_available(
                      ) and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      logger=[])

    if not cfg.general.test_only:
        if cfg.general.resume:
            print("config train method",cfg.general.train_method)
            model = DiscreteDenoisingDiffusion.load_from_checkpoint(cfg.general.resume,cfg=cfg,learning_rate=cfg.train.lr,
            amsgrad=cfg.train.amsgrad,weight_decay=cfg.train.weight_decay,**model_kwargs)
            # model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
            unfreeze_key = ["self_attn"]
            if len(unfreeze_key)>0 and cfg.general.partial:
                for name,param in model.named_parameters():
                    if "tf_layers" in name:
                        layernum  = int(name.split(".")[2])
                    else:
                        layernum = -1
                    if unfreeze_key[0] not in name or layernum<int(cfg.general.fix*cfg.model.n_layers):
                        param.requires_grad_(False)
                    else:
                        print(name)
                model.configure_optimizers()
            if cfg.dataset.name in ["zinc","moses"]:
                model.train_smiles = train_smiles
        else:
            if cfg.model.type == 'discrete':
                model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
            else:
                model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)
            unfreeze_key = ["self_attn"]
            if len(unfreeze_key)>0 and cfg.general.partial and cfg.general.train_method in ["ddpo","gdpo"] and "nodes" not in cfg.dataset:
                for name,param in model.named_parameters():
                    if "tf_layers" in name:
                        layernum  = int(name.split(".")[2])
                    else:
                        layernum = -1
                    if unfreeze_key[0] not in name or layernum<int(cfg.general.fix*cfg.model.n_layers):
                        param.requires_grad_(False)
                    else:
                        print(name)
                model.configure_optimizers()
            
            sd_dict = {"planar":home_prefix+"pretrained/planarpretrained.pt",
                        "sbm":home_prefix+"pretrained/sbmpretrained.pt",
                        "zinc":home_prefix+"pretrained/zincpretrained.pt",
                        "moses":home_prefix+"pretrained/mosespretrained.pt"}
            # sd_dict = {}
            print("batch size is {}".format(cfg.train.batch_size))
            if cfg.dataset.name in sd_dict and cfg.general.train_method in ["ddpo","gdpo"] and "nodes" not in cfg.dataset:
                sd = torch.load(sd_dict[cfg.dataset.name])
                new_sd = {}
                for k,v in sd.items():
                    if "model" in k:
                        new_sd[k[6:]]=v
                model.model.load_state_dict(new_sd)
                model.model.cuda()
                print("load pretrained model")
            print("load from check point")
            if cfg.dataset.name in ["zinc","moses"]:
                model.train_smiles = train_smiles
        init(cfg)
        trainer.fit(model, datamodule=datamodule)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        # cfg.general.test_method="evalproperty"
        cfg.general.test_method = "evalgeneral"
        if cfg.model.type == 'discrete':
            model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        else:
            model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)
        if cfg.dataset.name in ["zinc","moses"]:
            model.train_smiles = train_smiles
        model.ckpt = cfg.general.test_only
        trainer.test(model, datamodule=datamodule,
                     ckpt_path=cfg.general.test_only)
        # trainer.test(model, datamodule=datamodule)
        cfg.general.evaluate_all_checkpoints=False
        
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    # print("Loading checkpoint", ckpt_path)
                    global_step = torch.load(ckpt_path)["global_step"]
                    if global_step>400:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    setup_wandb(cfg)
                    model.ckpt = ckpt_path
                    trainer.test(model, datamodule=datamodule,
                                 ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
