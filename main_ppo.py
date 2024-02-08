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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import utils
from dataset.spectre_dataset import IMDBDataModule,SpectreDatasetInfos,PROTEINDataModule,MUTAGDataModule
from dataset.protein_dataset import ProteinDataModule, ProteinDatasetInfos, OGBHIVDataModule, OGBDatasetinfos, OGBPLUSDataModule,OGBHIVPOSDataModule, OGBHIVEXPOSDataModule
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics, ProteinSamplingMetrics, IMDBSamplingMetrics,MUTAGSamplingMetrics
from model.diffusion_discrete import DiscreteDenoisingDiffusion
from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from analysis.visualization import MolecularVisualization, NonMolecularVisualization
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures


warnings.filterwarnings("ignore", category=PossibleUserWarning)
batch_size = 128

def init(cfg):
    seed = cfg.general.seed
    seed = random.randint(0,10000)
    device = cfg.general.device
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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
    cfg.train.batch_size = batch_size
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
            resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(
            resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resumetwo'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    cfg.train.batch_size = 32
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
    dataset_config = cfg["dataset"]
    print(dataset_config)
    if dataset_config["name"] in ['planar', "MUTAG",'PROTEINS_full',"IMDB-BINARY"]:
        if dataset_config['name'] == 'plannar':
            datamodule = PlanarDataModule(cfg)
            sampling_metrics = PlanarSamplingMetrics(datamodule.dataloaders)
        elif dataset_config['name'] == 'protein':
            datamodule = ProteinDataModule(cfg)
            sampling_metrics = ProteinSamplingMetrics(datamodule.dataloaders)
        elif dataset_config["name"]=="IMDB-BINARY":
            datamodule = IMDBDataModule(cfg)
            sampling_metrics = IMDBSamplingMetrics(datamodule.dataloaders)
        elif dataset_config["name"]=="PROTEINS_full":
            datamodule = PROTEINDataModule(cfg)
            sampling_metrics = ProteinSamplingMetrics(datamodule.dataloaders)
        elif dataset_config["name"]=="MUTAG":
            datamodule = MUTAGDataModule(cfg)
            sampling_metrics = MUTAGSamplingMetrics(datamodule.dataloaders)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        # print(dataset_infos.node_types)
        train_metrics = TrainAbstractMetricsDiscrete(
        ) if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)
        print(dataset_infos)
        print("generate model kwargs")
        # print(dataset_infos.input_dims,dataset_infos.output_dims)
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    elif dataset_config["name"] in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-molppa", "ogbg-code2", "ogbg-molhivpos","ogbg-molplus", "ogbg-molhivexpos"]:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization
        if dataset_config["name"] == "ogbg-molhiv":
            datamodule = OGBHIVDataModule(cfg)
            datamodule.prepare_data()
            train_smiles = datamodule.graphs.train_smiles
            dataset_infos = OGBDatasetinfos(datamodule=datamodule, cfg=cfg)
        elif dataset_config["name"] == "ogbg-molplus":
            datamodule = OGBPLUSDataModule(cfg)
            datamodule.prepare_data()
            train_smiles = datamodule.graphs.train_smiles
            dataset_infos = OGBDatasetinfos(datamodule=datamodule, cfg=cfg)
        elif dataset_config["name"] == "ogbg-molhivpos":
            datamodule = OGBHIVPOSDataModule(cfg)
            datamodule.prepare_data()
            train_smiles = datamodule.graphs.train_smiles
            dataset_infos = OGBDatasetinfos(datamodule=datamodule, cfg=cfg)
        elif dataset_config["name"] == "ogbg-molhivexpos":
            datamodule = OGBHIVEXPOSDataModule(cfg)
            datamodule.prepare_data()
            train_smiles = datamodule.graphs.train_smiles
            dataset_infos = OGBDatasetinfos(datamodule=datamodule, cfg=cfg)
        print("data loaded")
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()
        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)
        
        # print(dataset_infos.input_dims)
        print("dataset_infos calculated")
        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)
            
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
        print("model_kwargs generated")
    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses', "zinc"]:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from dataset import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == "zinc":
            from dataset import zinc_dataset
            datamodule = zinc_dataset.MosesDataModule(cfg)
            dataset_infos = zinc_dataset.MOSESinfos(datamodule,cfg)
            train_smiles = pd.read_csv("./dataset/zinc/raw/zinc_train.csv")["smiles"].tolist()
        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
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
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
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
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule,
                     ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    setup_wandb(cfg)
                    trainer.test(model, datamodule=datamodule,
                                 ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
