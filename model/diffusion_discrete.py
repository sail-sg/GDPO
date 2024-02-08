import utils
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from metrics.train_metrics import TrainLossDiscrete
from analysis.rdkit_functions import compute_molecular_metrics_list,gen_smile_list
from scorer.evaluate import gen_score_list,gen_score_disc_list,gen_score_disc_listmose
from scorer.evaluate import evaluate as prop_evaluate
from scorer.evaluate import evaluatemose as mose_evaluate
from scorer.evaluate import evaluatelist as prop_df
from analysis.graphrewards import gen_reward_label as graph_labels
from analysis.graphrewards import gen_reward_list as graph_rewards
from analysis.graphrewards import gen_toy_reward_list as toy_rewards
from analysis.graphrewards import loader_to_nx
from diffusion import diffusion_utils
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from dataset.utils import smile2graph
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from model.transformer_model import GraphTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import random
import os
import sys
import json
from datetime import datetime
import numpy as np
o_path = os.getcwd()
sys.path.append(o_path)

from copy import deepcopy
from typing import Optional, Union, Dict, Any

from overrides import overrides
from pytorch_lightning.utilities import rank_zero_only

def to_sparse_batch(x, adj, mask=None):
    # transform x (B x N x D), adj (B x N x N), mask (B x N), here N is N_max
    # to x, edge_index, edge_attr/weight, batch

    B, N_max, D = x.shape
    # get num of nodes and reshape x
    num_nodes_graphs = torch.zeros_like(x[:,0,0], dtype=torch.int64).fill_(N_max)
    x = x.reshape(-1, D) # total_nodes * D

    # apply mask 
    if mask is not None:
        # mask adj
        adj = (adj * mask.unsqueeze(2)).transpose(1,2) 
        adj = (adj * mask.unsqueeze(2)).transpose(1,2) 
        # get number nodes per graph 
        num_nodes_graphs = mask.sum(dim=1)  # B
        # mask x
        x = x[mask.reshape(-1)] # total_nodes * D

    # get weight and index
    edge_weight = adj[adj.nonzero(as_tuple=True)]
    nnz_index = adj.nonzero().t()
    graph_idx, edge_index = nnz_index[0], nnz_index[1:]

    # get offset with size B
    offset_graphs = torch.cumsum(num_nodes_graphs, dim=0) # B
    offset_graphs = torch.cat([offset_graphs.new_zeros(1), offset_graphs]) # B+1
    
    # update edge index by adding offset
    edge_index += offset_graphs[graph_idx]

    # set up batch
    batch = torch.repeat_interleave(input=torch.arange(B, device=x.device), repeats=num_nodes_graphs )

    return x, edge_index, edge_weight, batch

def compress(X,E,node_mask):
    #E,(B,N,N,K)
    #X, (B,N,D)
    B = E.shape[0]
    N = E.shape[1]
    D = X.shape[2]
    new_X = torch.argmax(X,dim=-1)
    new_X = new_X.unsqueeze(-1)
    new_E = torch.argmax(E,dim=-1)
    # print("check compress input",X.shape,new_E.shape,node_mask.shape)
    x,edge_index,edge_weight,batch = to_sparse_batch(new_X,new_E,node_mask)
    return x,edge_index,edge_weight,batch



class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)
        self.lambda_train = self.cfg.model.lambda_train
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics
        
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features
        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(
                X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()

            x_marginals = node_types / torch.sum(node_types)
            

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(
                f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.train_round=0
        self.r_avg = 0
        self.r_std = 0
        self.validation_time = 0
        self.validation_set = {"0":[],"1":[]}
        workdir = os.getcwd()
        print("os working dir",workdir)
        if "multirun" in workdir:
            self.home_prefix = "./../../../../"
        else:
            self.home_prefix = "./../../../"
        self.update_time = 0
        self.train_smiles = None
        self.train_fps = None
        self.train_graphs = None
        self.ckpt = None
        if cfg.general.train_method in ["ddpo","gdpo"]:
            self.automatic_optimization=False

    def training_step(self, data, i):
        # method = self.cfg.general.train_method
        method = self.cfg.general.train_method
        if method == "orig":
            result = self.train_step_orig(data,i)
        elif method =="ddpo":
            result = self.train_step_ddpo(data,i)
        elif method=="gdpo":
            result = self.train_step_gdpo(data,i)
        return result

    def train_step_orig(self,data, i):
        if data.edge_index.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return
        # print("training step")
        # print(data.y)

        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        # print("the train shape of y",noisy_data["y_t"].shape,extra_data.y.shape)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # print(X.shape,E.shape)
        # sys.exit()
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    # def train_step_zppo(self,data,i):
    def write_reward(self,rewardavg,rewardstd):
        logfile = self.home_prefix+"train_log{}_new.log".format(self.cfg.dataset.name)
        if self.cfg.dataset.name in ["planar","sbm"] and "nodes" not in self.cfg.dataset:
            logfile = self.home_prefix+"train_log3.log"
        elif "nodes" in self.cfg.dataset:
            logfile = self.home_prefix+"train_log_toy_n{}.log".format(self.cfg.dataset.nodes)
            # self.log("val/epoch_score",rewardavg)
        logf = open(logfile,"a+")
        write_dict = {
            "lr":self.cfg.train.lr,
            "minibatchnorm":self.cfg.general.minibatchnorm,
            "batch_size":self.cfg.train.batch_size,
            "train_method":self.cfg.general.train_method,
            "dataset":self.cfg.dataset.name,
            "sampleloop":self.cfg.general.sampleloop,
            "step_freq":self.cfg.general.step_freq,
            "seed":self.cfg.general.seed,
            "interval":self.cfg.general.val_check_interval,
            "WD":self.cfg.train.weight_decay,
            "train_step":self.train_round,
            "rewardavg":round(rewardavg,6),
            "rewardstd":round(rewardstd,6),
            "SR":self.cfg.general.ppo_sr,
            "innerloop":self.cfg.general.innerloop,
            "partial":self.cfg.general.partial,
            "fix":self.cfg.general.fix
        }
        if self.cfg.dataset.name in ["zinc","moses"]:
            write_dict["target_prop"]=self.cfg.general.target_prop
            write_dict["discrete"]=self.cfg.general.discrete
            write_dict["thres"]=self.cfg.general.thres
        if "nodes" in self.cfg.dataset:
            write_dict["nodes"]=self.cfg.dataset.nodes
        # print(write_dict)
        line = json.dumps(write_dict)+"\n"
        logf.write(line)
        logf.close()
            
    def train_step_gdpo(self,data,i):
        #sampling
        opt = self.optimizers()
        bs = self.cfg.train.batch_size
        sample_list = []
        avgrewards = 0
        all_rewards = []
        gen_start = time.time()
        for _ in range(self.cfg.general.sampleloop):
            X_traj,E_traj,node_mask,rewards,rewardsmean = self.sample_batch_ppo(bs)
            # print(rewards)
            avgrewards+=rewardsmean
            all_rewards+=(rewards.tolist())
            X_now,E_now = X_traj[:-1],E_traj[:-1]
            X_0,E_0 = X_traj[-1],E_traj[-1]
            time_step = torch.Tensor([self.T-x for x in range(self.T)]).repeat(bs,1)
            #(T,bs)
            time_step = time_step.permute(1,0)
            # shuffle along time
            for idx in range(bs):
                perm = torch.randperm(self.T)
                X_now[:,idx,:,:],E_now[:,idx,:,:,:] = X_now[perm,idx,:,:],E_now[perm,idx,:,:,:]
                time_step[:,idx] = time_step[perm,idx]
            sample_list.append((X_now[:,:bs,:,:],E_now[:,:bs,:,:,:],X_0[:bs],E_0[:bs],time_step[:,:bs],node_mask[:bs],rewards[:bs]))
        gen_cost = time.time()-gen_start
        all_rewards = np.array(all_rewards)
        self.r_avg = all_rewards[all_rewards!=-1].mean()
        self.r_std = all_rewards[all_rewards!=-1].std()+1e-8
        # else:
        #     self.r_avg = 0.8*self.r_avg+0.2*all_rewards.mean()
        #     self.r_std = 0.8*self.r_std+0.2*(all_rewards.std()+1e-8)
        # print(all_rewards[all_rewards!=-1].mean())
        self.write_reward(self.r_avg,self.r_std)
        
        for loop_count in range(self.cfg.general.innerloop):
            opt.zero_grad()
            start_time = time.time()
            total_loss = 0
            pos_loss = 0
            neg_loss = 0
            pos_num = 0
            neg_num = 0
            pos_over = 0
            neg_over = 0
            for batch_idx in range(self.cfg.general.sampleloop):
                X_now,E_now,X_0,E_0,time_step,node_mask,rewards = sample_list[batch_idx]
                rewards_mask = rewards==-1
                pos_num += (rewards>0).sum().detach().cpu().numpy().item()
                neg_num += (rewards<=0).sum().detach().cpu().numpy().item()
                if self.cfg.general.minibatchnorm:
                    rewardsmean = (rewards[~rewards_mask]).mean()
                    rewardsstd = (rewards[~rewards_mask]).std()+1e-8
                else:
                    rewardsmean = self.r_avg
                    rewardsstd = self.r_std
                rewards = (rewards-rewardsmean)/rewardsstd
                pos_over += (rewards[~rewards_mask]>5).sum().detach().cpu().numpy().item()
                neg_over += (rewards[~rewards_mask]<-5).sum().detach().cpu().numpy().item()
                advantages = torch.clamp(rewards, -5, 5).cuda()
                advantages[rewards_mask]=0
                #accumulation on T steps
                X_0,E_0 = X_0.cuda(),E_0.cuda()
                sample_idx = random.sample(list(range(self.T)),int(self.T*self.cfg.general.ppo_sr))
                for idx in sample_idx:
                    X_t,E_t = X_now[idx],E_now[idx]
                    t_int = time_step[idx].reshape(bs,1)
                    y=torch.zeros(bs, 0)
                    t_float = t_int/self.T
                    s_float = (t_int-1)/self.T
                    t_float,s_float = t_float.cuda(),s_float.cuda()
                    X_t,E_t,y = X_t.cuda(),E_t.cuda(),y.cuda()
                    z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)
                    noisy_data = {'t': t_float, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
                    extra_data = self.compute_extra_data(noisy_data)
                    pred = self.forward(noisy_data, extra_data, node_mask)
                    # pred_X,pred_E = self.map_pred(X_t,E_t,pred,t_float,s_float,node_mask)
                    loss_X,loss_E = self.ppo_loss(masked_pred_X = pred.X,masked_pred_E=pred.E,pred_y=pred.y,true_X=X_0,true_E=E_0, true_y=y,reweight=advantages)
                    X_bs,E_bs = len(loss_X),len(loss_E)
                    pos_loss += (loss_X[loss_X>=0].sum()/X_bs+self.lambda_train[0]*loss_E[loss_E>=0].sum()/E_bs).detach().cpu().numpy().item()/(int(self.T*self.cfg.general.ppo_sr)*self.cfg.general.step_freq)
                    neg_loss += (loss_X[loss_X<0].sum()/X_bs+self.lambda_train[0]*loss_E[loss_E<0].sum()/E_bs).detach().cpu().numpy().item()/(int(self.T*self.cfg.general.ppo_sr)*self.cfg.general.step_freq)
                    loss = loss_X.mean() + self.lambda_train[0] * loss_E.mean()
                    loss = loss/(int(self.T*self.cfg.general.ppo_sr)*self.cfg.general.step_freq)
                    # print("train loss", loss)
                    self.manual_backward(loss)
                    total_loss+= loss.detach().cpu().numpy().item()
                if batch_idx%self.cfg.general.step_freq==self.cfg.general.step_freq-1:
                    self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                    opt.step()
                    opt.zero_grad()
            logfile = self.home_prefix+"profile_log_3.log"
            logf = open(logfile,"a+")
            time_cost = time.time()-start_time
            write_dict = {
            "lr":self.cfg.train.lr,
            "dataset":self.cfg.dataset.name,
            "batch_size":self.cfg.train.batch_size,
            "train_method": self.cfg.general.train_method,
            "sampleloop":self.cfg.general.sampleloop,
            "seed":self.cfg.general.seed,
            "WD":self.cfg.train.weight_decay,
            "train_step":self.train_round,
            "gen_cost":round(gen_cost,6),
            "train_loss":round(total_loss,6),
            "pos_loss": round(pos_loss,6),
            "neg_loss":round(neg_loss,6),
            "pos_num":pos_num,
            "neg_num":neg_num,
            "pos_over":pos_over,
            "neg_over":neg_over,
            "time_cost":round(time_cost,6),
            "loop_idx":loop_count,
            "SR":self.cfg.general.ppo_sr,
            "step_freq":self.cfg.general.step_freq,
            "innerloop":self.cfg.general.innerloop,
            "partial": self.cfg.general.partial,
            "fix":self.cfg.general.fix
            }
            if self.cfg.dataset.name in ["zinc","moses"]:
                write_dict["target_prop"]=self.cfg.general.target_prop
                write_dict["discrete"]=self.cfg.general.discrete
                write_dict["thres"]=self.cfg.general.thres
            # print(write_dict)
            line = json.dumps(write_dict)+"\n"
            logf.write(line)
            logf.close()
            # logf.close()
        self.train_round+=1
    
    def train_step_ddpo(self,data,i):
        #sampling
        # test_k1 = "model.tf_layers.0.self_attn.q.weight"
        # test_k2 = "model.mlp_in_X.0.weight"
        # for name,param in self.named_parameters():
        #     if name==test_k1:
        #         print(test_k1,param)
        #     if name==test_k2:
        #         print(test_k2,param)
        opt = self.optimizers()
        bs = self.cfg.train.batch_size
        sample_list = []
        avgrewards = 0
        all_rewards = []
        gen_start = time.time()
        for _ in range(self.cfg.general.sampleloop):
            X_traj,E_traj,node_mask,rewards,rewardsmean = self.sample_batch_ppo(bs)
            # print(rewards)
            avgrewards+=rewardsmean
            all_rewards+=(rewards.tolist())
            X_now,E_now = X_traj[:-1],E_traj[:-1]
            X_prev,E_prev = X_traj[1:],E_traj[1:]
            time_step = torch.Tensor([self.T-x for x in range(self.T)]).repeat(bs,1)
            #(T,bs)
            time_step = time_step.permute(1,0)
            # shuffle along time
            for idx in range(bs):
                perm = torch.randperm(self.T)
                X_now[:,idx,:,:],E_now[:,idx,:,:,:] = X_now[perm,idx,:,:],E_now[perm,idx,:,:,:]
                X_prev[:,idx,:,:],E_prev[:,idx,:,:,:] = X_prev[perm,idx,:,:],E_prev[perm,idx,:,:,:]
                time_step[:,idx] = time_step[perm,idx]
            sample_list.append((X_now[:,:bs,:,:],E_now[:,:bs,:,:,:],X_prev[:,:bs,:,:],E_prev[:,:bs,:,:,:],time_step[:,:bs],node_mask[:bs],rewards[:bs]))
        gen_cost = time.time()-gen_start
        all_rewards = np.array(all_rewards)
        self.r_avg = all_rewards[all_rewards!=-1].mean()
        self.r_std = all_rewards[all_rewards!=-1].std()+1e-8
        # else:
        #     self.r_avg = 0.8*self.r_avg+0.2*all_rewards.mean()
        #     self.r_std = 0.8*self.r_std+0.2*(all_rewards.std()+1e-8)
        # print(all_rewards[all_rewards!=-1].mean())
        self.write_reward(self.r_avg,self.r_std)
        logfile = self.home_prefix+"profile_log3.log"
        logf = open(logfile,"a+")
        for loop_count in range(self.cfg.general.innerloop):
            opt.zero_grad()
            start_time = time.time()
            total_loss = 0
            pos_loss = 0
            neg_loss = 0
            pos_num = 0
            neg_num = 0
            pos_over = 0
            neg_over = 0
            for batch_idx in range(self.cfg.general.sampleloop):
                X_now,E_now,X_prev,E_prev,time_step,node_mask,rewards = sample_list[batch_idx]
                rewards_mask = rewards==-1
                pos_num += (rewards>0).sum().detach().cpu().numpy().item()
                neg_num += (rewards<=0).sum().detach().cpu().numpy().item()
                if self.cfg.general.minibatchnorm:
                    rewardsmean = (rewards[~rewards_mask]).mean()
                    rewardsstd = (rewards[~rewards_mask]).std()+1e-8
                else:
                    rewardsmean = self.r_avg
                    rewardsstd = self.r_std
                rewards = (rewards-rewardsmean)/rewardsstd
                pos_over += (rewards[~rewards_mask]>5).sum().detach().cpu().numpy().item()
                neg_over += (rewards[~rewards_mask]<-5).sum().detach().cpu().numpy().item()
                advantages = torch.clamp(rewards, -5, 5).cuda()
                advantages[rewards_mask]=0
                #accumulation on T steps
                sample_idx = random.sample(list(range(self.T)),int(self.T*self.cfg.general.ppo_sr))
                for idx in sample_idx:
                    X_t,E_t = X_now[idx],E_now[idx]
                    t_int = time_step[idx].reshape(bs,1)
                    y=torch.zeros(bs, 0)
                    t_float = t_int/self.T
                    s_float = (t_int-1)/self.T
                    t_float,s_float = t_float.cuda(),s_float.cuda()
                    X_t,E_t,y = X_t.cuda(),E_t.cuda(),y.cuda()
                    z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)
                    noisy_data = {'t': t_float, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
                    extra_data = self.compute_extra_data(noisy_data)
                    pred = self.forward(noisy_data, extra_data, node_mask)
                    pred_X,pred_E = self.map_pred(X_t,E_t,pred,t_float,s_float,node_mask)

                    X_t,E_t = X_prev[idx],E_prev[idx]
                    X_t,E_t = X_t.cuda(),E_t.cuda()
                    loss_X,loss_E = self.nll_loss(masked_pred_X = pred_X,masked_pred_E=pred_E,pred_y=pred.y,true_X=X_t,true_E=E_t, true_y=y,reweight=advantages)
                    # print(loss_X,loss_E)
                    X_bs,E_bs = len(loss_X),len(loss_E)
                    pos_loss += (loss_X[loss_X>=0].sum()/X_bs+self.lambda_train[0]*loss_E[loss_E>=0].sum()/E_bs).detach().cpu().numpy().item()/(int(self.T*self.cfg.general.ppo_sr)*self.cfg.general.step_freq)
                    neg_loss += (loss_X[loss_X<0].sum()/X_bs+self.lambda_train[0]*loss_E[loss_E<0].sum()/E_bs).detach().cpu().numpy().item()/(int(self.T*self.cfg.general.ppo_sr)*self.cfg.general.step_freq)
                    loss = loss_X.mean() + self.lambda_train[0] * loss_E.mean()
                    loss = loss/(int(self.T*self.cfg.general.ppo_sr)*self.cfg.general.step_freq)
                    # print("train loss", loss)
                    self.manual_backward(loss)
                    total_loss+= loss.detach().cpu().numpy().item()
                if batch_idx%self.cfg.general.step_freq==self.cfg.general.step_freq-1:
                    self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                    opt.step()
                    opt.zero_grad()
            time_cost = time.time()-start_time
            write_dict = {
            "lr":self.cfg.train.lr,
            "dataset":self.cfg.dataset.name,
            "batch_size":self.cfg.train.batch_size,
            "sampleloop":self.cfg.general.sampleloop,
            "seed":self.cfg.general.seed,
            "WD":self.cfg.train.weight_decay,
            "train_step":self.train_round,
            "gen_cost":round(gen_cost,6),
            "train_loss":round(total_loss,6),
            "pos_loss": round(pos_loss,6),
            "neg_loss":round(neg_loss,6),
            "pos_num":pos_num,
            "neg_num":neg_num,
            "pos_over":pos_over,
            "neg_over":neg_over,
            "time_cost":round(time_cost,6),
            "loop_idx":loop_count,
            "SR":self.cfg.general.ppo_sr,
            "step_freq":self.cfg.general.step_freq,
            "innerloop":self.cfg.general.innerloop,
            "partial": self.cfg.general.partial,
            "fix":self.cfg.general.fix
            }
            if self.cfg.dataset.name in ["zinc","moses"]:
                write_dict["target_prop"]=self.cfg.general.target_prop
                write_dict["discrete"]=self.cfg.general.discrete
                write_dict["thres"]=self.cfg.general.thres
            # print(write_dict)
            line = json.dumps(write_dict)+"\n"
            logf.write(line)
        self.train_round+=1

    def map_pred(self,X_t,E_t,pred,t,s,node_mask):
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        # bs, n, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
        unnormalized_prob_X = weighted_X.sum(
            dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / \
            torch.sum(unnormalized_prob_X, dim=-1,
                      keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        # bs, N, d0, d_t-1
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / \
            torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
        prob_Z = utils.PlaceHolder(X=prob_X, E=prob_E, y=None).mask(node_mask)
        return prob_Z.X,prob_Z.E
    def ppo_loss(self,masked_pred_X,masked_pred_E,pred_y,true_X,true_E,true_y,reweight):
        #reweight #(bs)
        # print("check loss shape")
        # print(masked_pred_X.shape,masked_pred_E.shape,true_X.shape,true_E.shape,reweight.shape)
        b,n,_ = true_X.shape
        reweight = reweight.reshape(b,1)
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)
        metric = nn.CrossEntropyLoss(reduction="none")
        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[:, :]
        flat_true_X = torch.argmax(flat_true_X, dim=-1)
        flat_pred_X = masked_pred_X[:, :]
        loss_X = metric(flat_pred_X,flat_true_X)
        loss_X = loss_X.reshape(b,-1)*reweight
        loss_X = loss_X.view(-1)
        loss_X = loss_X[mask_X]
        flat_true_E = true_E[:, :]
        flat_true_E = torch.argmax(flat_true_E, dim=-1)
        flat_pred_E = masked_pred_E[:, :]
        loss_E = metric(flat_pred_E, flat_true_E)
        loss_E = loss_E.reshape(b,-1)*reweight
        loss_E = loss_E.view(-1)
        loss_E = loss_E[mask_E]
        
        return loss_X,loss_E
    def nll_loss(self,masked_pred_X,masked_pred_E,pred_y,true_X,true_E,true_y,reweight):
        #reweight #(bs)
        # print("check loss shape")
        # print(masked_pred_X.shape,masked_pred_E.shape,true_X.shape,true_E.shape,reweight.shape)
        b,n,_ = true_X.shape
        reweight = reweight.reshape(b,1)
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)
        # metric = nn.CrossEntropyLoss(reduction="none")
        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[:, :]
        flat_true_X = torch.argmax(flat_true_X, dim=-1)
        flat_pred_X = masked_pred_X[:, :]
        flat_true_X = F.one_hot(flat_true_X, num_classes=flat_pred_X.shape[-1]).float()
        # loss_X = metric(flat_pred_X,flat_true_X)
        # print(flat_pred_X.shape,flat_true_X.shape)
        loss_X = -torch.log((flat_pred_X*flat_true_X).sum(-1))
        # print(loss_X,"X loss nll")
        loss_X = loss_X.reshape(b,-1)*reweight
        loss_X = loss_X.view(-1)
        loss_X = loss_X[mask_X]
        flat_true_E = true_E[:, :]
        flat_true_E = torch.argmax(flat_true_E, dim=-1)
        flat_pred_E = masked_pred_E[:, :]
        flat_true_E = F.one_hot(flat_true_E, num_classes=flat_pred_E.shape[-1]).float()
        # loss_E = metric(flat_pred_E, flat_true_E)
        loss_E = -torch.log((flat_pred_E*flat_true_E).sum(-1))
        # print(loss_E,"E loss nll")
        loss_E = loss_E.reshape(b,-1)*reweight
        loss_E = loss_E.view(-1)
        loss_E = loss_E[mask_E]
        
        return loss_X,loss_E

    def logp(self,masked_pred_X,masked_pred_E,pred_y,true_X,true_E,true_y):
        #reweight #(bs)
        # print("check loss shape")
        # print(masked_pred_X.shape,masked_pred_E.shape,true_X.shape,true_E.shape,reweight.shape)
        b,n,_ = true_X.shape
        # reweight = reweight.reshape(b,1)
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)
        metric = nn.CrossEntropyLoss(reduction="none")
        # Remove masked rows

        flat_true_X = true_X[:, :]
        flat_true_X = torch.argmax(flat_true_X, dim=-1)
        flat_pred_X = masked_pred_X[:, :]
        loss_X = metric(flat_pred_X,flat_true_X)
        loss_X = loss_X.reshape(b,-1)
        flat_true_E = true_E[:, :]
        flat_true_E = torch.argmax(flat_true_E, dim=-1)
        flat_pred_E = masked_pred_E[:, :]
        loss_E = metric(flat_pred_E, flat_true_E)
        loss_E = loss_E.reshape(b,-1)
        
        return -loss_X,-loss_E


    def configure_optimizers(self):
        pg = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(pg, lr=self.cfg.train.lr, amsgrad=self.cfg.train.amsgrad,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        if self.trainer.datamodule is not None:
            self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        else:
            print(dir(self.trainer))
            self.train_iterations = len(self.trainer.train_dataloader)
        print("Size of the input features", self.Xdim, self.Edim, self.ydim)

    def on_train_epoch_start(self) -> None:
        print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_loss.log_epoch_metrics(
            self.current_epoch, self.start_epoch_time)
        self.train_metrics.log_epoch_metrics(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    @torch.no_grad()
    def validation_step(self, data, i):
        if self.cfg.general.val_method=="orig":
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            # print("validation step", dense_data.X.shape,dense_data.E.shape)
            noisy_data = self.apply_noise(
                dense_data.X, dense_data.E, data.y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            # print("the validation shape of y is ", noisy_data["y_t"].shape,extra_data.y.shape)

            pred = self.forward(noisy_data, extra_data, node_mask)
            nll = self.compute_val_loss(
                pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)
        else:
            pass
            nll=0
        return {'loss': nll}
    @torch.no_grad()
    def ppo_evaluate(self,test=False):
        # samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        if self.cfg.general.vallike:
            pos_loss = 0
            for X,E,node_mask in self.validation_set["1"]:
                bs = X.shape[0]
                y=torch.zeros(bs, 0)
                X,E,y = X.cuda(),E.cuda(),y.cuda()
                node_mask = node_mask.cuda()
                z_t = utils.PlaceHolder(X=X, E=E, y=y).type_as(X).mask(node_mask)
                noisy_data = self.apply_noise(z_t.X,z_t.E,z_t.y,node_mask)
                extra_data = self.compute_extra_data(noisy_data)
                pred = self.forward(noisy_data, extra_data, node_mask)
                nll = self.compute_val_loss(
                pred, noisy_data, z_t.X, z_t.E, z_t.y,  node_mask, test=False,no_prior=True)
                pos_loss += nll.detach().cpu().numpy().item()
            pos_loss = pos_loss/(len(self.validation_set["1"])+1e-8)
            neg_loss = 0
            for X,E,node_mask in self.validation_set["0"]:
                bs = X.shape[0]
                y=torch.zeros(bs, 0)
                X,E,y = X.cuda(),E.cuda(),y.cuda()
                node_mask = node_mask.cuda()
                z_t = utils.PlaceHolder(X=X, E=E, y=y).type_as(X).mask(node_mask)
                noisy_data = self.apply_noise(z_t.X,z_t.E,z_t.y,node_mask)
                extra_data = self.compute_extra_data(noisy_data)
                pred = self.forward(noisy_data, extra_data, node_mask)
                nll = self.compute_val_loss(
                pred, noisy_data, z_t.X, z_t.E, z_t.y,  node_mask, test=False,no_prior=True)
                neg_loss += nll.detach().cpu().numpy().item()
            neg_loss = neg_loss/(len(self.validation_set["0"])+1e-8)
            self.log("val/epoch_score", neg_loss-pos_loss)
            logfile = self.home_prefix+"/validation_dict.log"
            logf = open(logfile,"a+")
            write_dict = {
                "lr":self.cfg.train.lr,
                "dataset":self.cfg.dataset.name,
                "SR":self.cfg.general.ppo_sr,
                "step_freq":self.cfg.general.step_freq,
                "sampleloop":self.cfg.general.sampleloop,
                "seed":self.cfg.general.seed,
                "interval":self.cfg.general.val_check_interval,
                "WD":self.cfg.train.weight_decay,
                "train_step":self.train_round,
                "pos_like":round(pos_loss,6),
                "neg_like":round(neg_loss,6),
                "minibatchnorm":self.cfg.general.minibatchnorm,
                "train_method":self.cfg.general.train_method,
                "innerloop":self.cfg.general.innerloop
            }
            line = json.dumps(write_dict)+"\n"
            logf.write(line)
        else:
            if self.cfg.dataset.name in ["planar","sbm"] and "nodes" not in self.cfg.dataset:
                samples_left_to_generate = 64
            elif "nodes" in self.cfg.dataset:
                samples_left_to_generate = 2048
            else:
                samples_left_to_generate = 512
            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save
            samples = []
            id = 0
            while samples_left_to_generate > 0:
                print(f'Samples left to generate: {samples_left_to_generate}/'
                    f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                                keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
                id += to_generate
                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            if self.cfg.dataset.name in ["zinc","moses"]:
                if self.train_fps is None:
                    assert self.train_smiles is not None
                    train_mols = [Chem.MolFromSmiles(smi) for smi in self.train_smiles]
                    self.train_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in train_mols]
                smiles,valid_r,uniq,uniq_r,freq_list = gen_smile_list(samples,self.dataset_info)
                valid = [x for x in smiles if x is not None]
                if self.cfg.dataset.name == "moses":
                    result = mose_evaluate(self.cfg.general.target_prop,valid,None,self.train_fps)
                else:
                    result = prop_evaluate(self.cfg.general.target_prop,valid,None,self.train_fps)
                if self.cfg.dataset.name == "zinc":
                    logfile = self.home_prefix+"evaluation_dictzinc.log"
                elif self.cfg.dataset.name == "moses":
                    logfile = self.home_prefix+"evaluation_dictmoses.log"
                if test:
                    logfile = self.home_prefix+"test_property_result.log"
                    logf = open(logfile,"a+")
                    print(result)
                    write_dict = {
                        # "time":time_stamp,
                        "seed":self.cfg.general.seed,
                        "dataset":self.cfg.dataset.name,
                        "target_prop":self.cfg.general.target_prop,
                        "VALID":round(100*valid_r,4),
                        "UNIQ":round(100*uniq_r,4),
                        "novelty":result["novelty"],
                        "top_ds":result["top_ds"],
                        "avgscore":result["avgscore"],
                        "hit":result["hit"],
                        "avgds": round(result["avgds"],4),
                        "avgqed": round(result["avgqed"],4),
                        "avgsa": round(result["avgsa"],4)
                    }
                    line = json.dumps(write_dict)+"\n"
                    logf.write(line)
                    logf.close()
                else:
                    logf = open(logfile,"a+")
                    write_dict = {
                        "lr":self.cfg.train.lr,
                        "batch_size":self.cfg.train.batch_size,
                        "SR":self.cfg.general.ppo_sr,
                        "dataset":self.cfg.dataset.name,
                        "step_freq":self.cfg.general.step_freq,
                        "target_prop":self.cfg.general.target_prop,
                        "sampleloop":self.cfg.general.sampleloop,
                        "seed":self.cfg.general.seed,
                        "interval":self.cfg.general.val_check_interval,
                        "WD":self.cfg.train.weight_decay,
                        "valround":self.validation_time,
                        "VALID":round(100*valid_r,4),
                        "UNIQ":round(100*uniq_r,4),
                        "amsgrad":self.cfg.train.amsgrad,
                        "EMA":self.cfg.train.ema_decay,
                        "train_method":self.cfg.general.train_method,
                        "weight_list": "q{}_s{}_n{}_d{}".format(*self.cfg.general.weight_list),
                        "innerloop":self.cfg.general.innerloop,
                        "partial":self.cfg.general.partial,
                        "fix":self.cfg.general.fix,
                        "novelty":result["novelty"],
                        "top_ds":result["top_ds"],
                        "avgscore":result["avgscore"],
                        "hit":result["hit"],
                        "avgds": round(result["avgds"],4),
                        "avgqed": round(result["avgqed"],4),
                        "avgsa": round(result["avgsa"],4)
                    }
                    write_dict["discrete"]=self.cfg.general.discrete
                    write_dict["thres"]=self.cfg.general.thres
                    self.log("val/epoch_score", 100*result["hit"] if valid_r>0.2 else 0)
                    line = json.dumps(write_dict)+"\n"
                    logf.write(line)
                    self.validation_time += 1
            elif self.cfg.dataset.name in ["planar","sbm"] and "nodes" not in self.cfg.dataset:
                res = self.sampling_metrics(
                    samples, self.cfg.dataset.name, self.current_epoch, val_counter=-1, test=False)
                self.sampling_metrics.reset()
                logfile = self.home_prefix+r"evaluation_dict4.log"
                logf = open(logfile,"a+")
                write_dict = {
                    "lr":self.cfg.train.lr,
                    "SR":self.cfg.general.ppo_sr,
                    "train_method": self.cfg.general.train_method,
                    "batch_size":self.cfg.train.batch_size,
                    "dataset":self.cfg.dataset.name,
                    "step_freq":self.cfg.general.step_freq,
                    "sampleloop":self.cfg.general.sampleloop,
                    "seed":self.cfg.general.seed,
                    "interval":self.cfg.general.val_check_interval,
                    "WD":self.cfg.train.weight_decay,
                    "valround":self.validation_time,
                    "amsgrad":self.cfg.train.amsgrad,
                    "EMA":self.cfg.train.ema_decay,
                    "train_method":self.cfg.general.train_method,
                    "innerloop":self.cfg.general.innerloop,
                    "partial":self.cfg.general.partial,
                    "degree":round(res["degree"],4),
                    "clustering":round(res["clustering"],4),
                    "orbit":round(res["orbit"],4),
                    "non_iso":round(res["sampling/frac_non_iso"],4),
                    "uniq_frac":round(res["sampling/frac_unique"],4)
                }
                if self.cfg.dataset.name =="planar":
                    write_dict["planar_acc"]= round(res["planar_acc"],4)
                    avgscore = res["planar_acc"]
                else:
                    write_dict["sbm_acc"]= round(res["sbm_acc"],4)
                    avgscore = res["sbm_acc"]
                self.log("val/epoch_score",avgscore)
                line = json.dumps(write_dict)+"\n"
                logf.write(line)
                self.validation_time += 1
            elif "nodes" in self.cfg.dataset:
                res = toy_rewards(samples)
                logfile = self.home_prefix+"evaluation_toy_n{}.log".format(self.cfg.dataset.nodes)
                avgreward = np.array(res).mean()
                logf = open(logfile,"a+")
                avgscore = round(100*np.array(res).sum()/len(res),4)
                write_dict = {
                    "lr":self.cfg.train.lr,
                    "SR":self.cfg.general.ppo_sr,
                    "train_method": self.cfg.general.train_method,
                    "batch_size":self.cfg.train.batch_size,
                    "dataset":self.cfg.dataset.name,
                    "step_freq":self.cfg.general.step_freq,
                    "sampleloop":self.cfg.general.sampleloop,
                    "seed":self.cfg.general.seed,
                    "interval":self.cfg.general.val_check_interval,
                    "valround":self.validation_time,
                    "train_method":self.cfg.general.train_method,
                    "innerloop":self.cfg.general.innerloop,
                    "avgreward":round(avgreward,6)
                }
                print(write_dict)
                self.log("val/epoch_score",avgscore)
                line = json.dumps(write_dict)+"\n"
                logf.write(line)
                logf.close()
                self.validation_time += 1
            print("Done.")
    def validation_epoch_end(self, outs) -> None:
        if self.cfg.general.val_method=="orig":
            metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                    self.val_X_logp.compute(), self.val_E_logp.compute()]
            wandb.log({"val/epoch_NLL": metrics[0],
                    "val/X_kl": metrics[1],
                    "val/E_kl": metrics[2],
                    "val/X_logp": metrics[3],
                    "val/E_logp": metrics[4]}, commit=False)

            print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                f"Val Edge type KL: {metrics[2] :.2f}")

            # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
            val_nll = metrics[0]
            self.log("val/epoch_NLL", val_nll)

            if val_nll < self.best_val_nll:
                self.best_val_nll = val_nll
            print('Val loss: %.4f \t Best val loss:  %.4f\n' %
                (val_nll, self.best_val_nll))

            self.val_counter += 1
            if self.val_counter % self.cfg.general.sample_every_val == 0:
                start = time.time()
                samples_left_to_generate = self.cfg.general.samples_to_generate
                samples_left_to_save = self.cfg.general.samples_to_save
                chains_left_to_save = self.cfg.general.chains_to_save

                samples = []

                ident = 0
                while samples_left_to_generate > 0:
                    bs = 2 * self.cfg.train.batch_size
                    to_generate = min(samples_left_to_generate, bs)
                    to_save = min(samples_left_to_save, bs)
                    chains_save = min(chains_left_to_save, bs)
                    samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                    save_final=to_save,
                                                    keep_chain=chains_save,
                                                    number_chain_steps=self.number_chain_steps))
                    ident += to_generate

                    samples_left_to_save -= to_save
                    samples_left_to_generate -= to_generate
                    chains_left_to_save -= chains_save
                print("Computing sampling metrics...")
                self.sampling_metrics(
                    samples, self.name, self.current_epoch, val_counter=-1, test=False)
                print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
                self.sampling_metrics.reset()
        elif self.cfg.general.val_method=="ppo":
            self.ppo_evaluate()
        else:
            pass

    def on_test_epoch_start(self) -> None:
        print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()

    def test_step(self, data, i):
        if self.cfg.general.test_method=="orig":
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            noisy_data = self.apply_noise(
                dense_data.X, dense_data.E, data.y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            # print("the test shape of y is ",noisy_data["y_t"].shape,extra_data.y.shape)
            pred = self.forward(noisy_data, extra_data, node_mask)
            nll = self.compute_val_loss(
                pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
            return {'loss': nll}
        else:
            return {"loss":0}
    def test_epoch_end(self,outs):
        # method = self.cfg.general.test_method
        method = self.cfg.general.test_method
        if method=="orig":
            print("sample with original method")
            self.test_epoch_end_orig(outs)
        elif method == "ppo":
            print("sample trace for ppo")
            self.test_epoch_end_ppo(outs)
        elif method == "olppo":
            print("likelihood and generation")
            self.test_epoch_end_olppo(outs)
        elif method == "general":
            print("likelihood and generation")
            self.test_epoch_end_general(outs)
        elif method == "evalgeneral":
            print("eval general graph model")
            self.test_epoch_end_evalgeneral(outs)
        elif method == "evalproperty":
            print("eval molecule property")
            self.test_epoch_end_evalproperty(outs)
            # seed= 431
            # if seed is not None:
            #     torch.manual_seed(seed)
            #     torch.cuda.manual_seed(seed)
            #     np.random.seed(seed)
            #     random.seed(seed)
            # self.ppo_evaluate(test=True)
        else:
            raise ValueError

    def test_epoch_end_olppo(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        # metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
        #            self.test_X_logp.compute(), self.test_E_logp.compute()]
        
        smile_path = {
            "parp1": r"./dataset/zinc/raw/parp1.txt",
            "fa7":r"./dataset/zinc/raw/fa7.txt",
            "5ht1b":r"./dataset/zinc/raw/5ht1b.txt",
            "braf":r"./dataset/zinc/raw/braf.txt",
            "jak2":r"./dataset/zinc/raw/jak2.txt"
        }
        print("test prop is {}".format(self.cfg.general.target_prop))
        if self.train_fps is None:
            assert self.train_smiles is not None
            train_mols = [Chem.MolFromSmiles(smi) for smi in self.train_smiles]
            self.train_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in train_mols]
        smile_list = []
        with open(smile_path[self.cfg.general.target_prop],"r") as f:
            for line in f:
                smi = line.strip().split()[0]
                smile_list.append(smi)
        f.close()
        ratio = 0.2
        test_loader = smile2graph(smile_list[:int(ratio*len(smile_list))],self.cfg.train.batch_size)
        NLL = 0
        sample_num = 0
        for data in test_loader:
            dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            # X, E = X.cuda(),E.cuda()
            bs = X.shape[0]
            y=torch.zeros(bs, 0)
            X,E,y = X.cuda(),E.cuda(),y.cuda()
            node_mask = node_mask.cuda()
            z_t = utils.PlaceHolder(X=X, E=E, y=y).type_as(X).mask(node_mask)
            noisy_data = self.apply_noise(z_t.X,z_t.E,z_t.y,node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            nll = self.compute_val_loss(
            pred, noisy_data, z_t.X, z_t.E, z_t.y,  node_mask, test=False,no_prior=True)
            NLL += nll.detach().cpu().numpy().item()
        logf=r"./test_log2.log"
        NLL = NLL/len(test_loader)
        logf = open(logf,"a+")
        resume_path = self.ckpt
        global_step = torch.load(resume_path)["global_step"]
        write_dict = {
            "resume":self.ckpt,
            "NLL":"{:.4f}".format(NLL),
            "step":global_step
        }
        line = json.dumps(write_dict)+"\n"
        logf.write(line)
        logf.close()
    
    def test_epoch_end_orig(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        wandb.log({"test/epoch_NLL": metrics[0],
                   "test/X_kl": metrics[1],
                   "test/E_kl": metrics[2],
                   "test/X_logp": metrics[3],
                   "test/E_logp": metrics[4]}, commit=False)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
              f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        print(f'Test loss: {test_nll :.4f}')

        # samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_generate = 256
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        start = time.time()
        while samples_left_to_generate > 0:
            print(f'Samples left to generate: {samples_left_to_generate}/'
                  f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        time_cost = time.time()-start
        print("Saving the generated graphs")
        filename = "generated_samples_orig_t{:.4f}num{}.txt".format(time_cost/1000,len(samples))
        if os.path.exists(filename):
            filename = filename[:-4]+"d.txt"
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        print("Saved.")
        print("Computing sampling metrics...")
        self.sampling_metrics.reset()
        self.sampling_metrics(samples, self.name,
                              self.current_epoch, self.val_counter, test=True)
        self.sampling_metrics.reset()
        print("Done.")

    def test_epoch_end_general(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        wandb.log({"test/epoch_NLL": metrics[0],
                   "test/X_kl": metrics[1],
                   "test/E_kl": metrics[2],
                   "test/X_logp": metrics[3],
                   "test/E_logp": metrics[4]}, commit=False)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
              f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        print(f'Test loss: {test_nll :.4f}')

        # samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_generate = 1024
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        start = time.time()
        seed= 666
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        while samples_left_to_generate > 0:
            print(f'Samples left to generate: {samples_left_to_generate}/'
                  f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        time_cost = time.time()-start
        label = graph_labels(samples,self.cfg.dataset.name)
        filename = "generated_samples_orig_num{}_seed{}.txt".format(len(samples),seed)
        labelname = "generated_label_orig_num{}_seed{}.txt".format(len(samples),seed)
        with open(labelname,"w") as f:
            line = json.dumps(label)
            f.write(line+"\n")
        f.close()
        print("Saving the generated graphs")
        # filename = "generated_samples_orig_num{}_seed{}.txt".format(len(samples),self.cfg.general.seed)
        if os.path.exists(filename):
            filename = filename[:-4]+"d.txt"
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        print("Saved.")
        # print("Computing sampling metrics...")
        # self.sampling_metrics.reset()
        # self.sampling_metrics(samples, self.name,
        #                       self.current_epoch, self.val_counter, test=True)
        # self.sampling_metrics.reset()
        # print("Done.")
    @torch.no_grad()
    def test_epoch_end_evalproperty(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        self.model.eval()
        # samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_generate = 2048
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        start = time.time()
        seed= 6666
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save
        samples = []
        id = 0
        while samples_left_to_generate > 0:
            print(f'Samples left to generate: {samples_left_to_generate}/'
                f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                            keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        filename = "generated_samples_orig_num{}_seed{}_prop{}.txt".format(len(samples),seed,self.cfg.general.target_prop)
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        print("Saved.")
        sperate = 512
        K = len(samples)//sperate
        #train-test base perform
        time_stamp = "{}".format(datetime.now())
        if self.train_fps is None:
            print("prepare train fps")
            assert self.train_smiles is not None
            train_mols = [Chem.MolFromSmiles(smi) for smi in self.train_smiles]
            self.train_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in train_mols]
        logfile = self.home_prefix+"test_property_result.log"
        for idx in range(K):
            print("start eval idx {}".format(idx))
            subsamples = samples[idx*sperate:min((idx+1)*sperate,len(samples))]
            print(self.dataset_info)
            smiles,valid_r,uniq,uniq_r,freq_list = gen_smile_list(subsamples,self.dataset_info)
            valid = [x for x in smiles if x is not None]
            print(self.cfg.dataset.name,"dataset name is ")
            if self.cfg.dataset.name == "moses":
                result = mose_evaluate(self.cfg.general.target_prop,valid,None,self.train_fps)
            else:
                result = prop_evaluate(self.cfg.general.target_prop,valid,None,self.train_fps)
            logf = open(logfile,"a+")
            print(result)
            write_dict = {
                "time":time_stamp,
                "dataset":self.cfg.dataset.name,
                "step_freq":self.cfg.general.step_freq,
                "target_prop":self.cfg.general.target_prop,
                "VALID":round(100*valid_r,4),
                "UNIQ":round(100*uniq_r,4),
                "novelty":result["novelty"],
                "top_ds":result["top_ds"],
                "avgscore":result["avgscore"],
                "hit":result["hit"],
                "avgds": round(result["avgds"],4),
                "avgqed": round(result["avgqed"],4),
                "avgsa": round(result["avgsa"],4)
            }
            smile_path = "sample_smi"
            line = json.dumps(write_dict)+"\n"
            logf.write(line)
            logf.close()
    
    def test_epoch_end_evalgeneral(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        wandb.log({"test/epoch_NLL": metrics[0],
                   "test/X_kl": metrics[1],
                   "test/E_kl": metrics[2],
                   "test/X_logp": metrics[3],
                   "test/E_logp": metrics[4]}, commit=False)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
              f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        print(f'Test loss: {test_nll :.4f}')

        # samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        if self.cfg.dataset.name == "sbm":
            samples_left_to_generate = 512
        elif self.cfg.dataset.name == "planar":
            samples_left_to_generate = 512
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        start = time.time()
        seed= 666
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        while samples_left_to_generate > 0:
            print(f'Samples left to generate: {samples_left_to_generate}/'
                  f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        time_cost = time.time()-start
        # label = graph_labels(samples,self.cfg.dataset.name)
        filename = "generated_samples_orig_num{}_seed{}.txt".format(len(samples),seed)
        # labelname = "generated_label_orig_num{}_seed{}.txt".format(len(samples),seed)
        # with open(labelname,"w") as f:
        #     line = json.dumps(label)
        #     f.write(line+"\n")
        # f.close()
        print("Saving the generated graphs")
        # filename = "generated_samples_orig_num{}_seed{}.txt".format(len(samples),self.cfg.general.seed)
        if os.path.exists(filename):
            filename = filename[:-4]+"d.txt"
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        print("Saved.")

        res = self.sampling_metrics.testperf()
        logfile = self.home_prefix+"test_result.log"
        logf = open(logfile,"a+")
        write_dict = {
            "dataname":self.cfg.dataset.name,
            "testperf": True,
            "degree":round(res["degree"],6),
            "clustering":round(res["clustering"],6),
            "orbit":round(res["orbit"],6),
        }
        line = json.dumps(write_dict)+"\n"
        logf.write(line)
        logf.close()

        if self.cfg.dataset.name == "sbm":
            sperate = 128
        elif self.cfg.dataset.name == "planar":
            sperate = 128
        K = len(samples)//sperate
        #train-test base perform
        time_stamp = "{}".format(datetime.now())
        for idx in range(K):
            print("start eval idx {}".format(idx))
            subsamples = samples[idx*sperate:min((idx+1)*sperate,len(samples))]
            res = self.sampling_metrics(
                    subsamples, self.cfg.dataset.name, self.current_epoch, val_counter=-1, test=True)
            # print(res)
            self.sampling_metrics.reset()
            logfile = self.home_prefix+"test_result.log"
            logf = open(logfile,"a+")
            write_dict = {
                "time": time_stamp,
                "train_method":self.cfg.general.train_method,
                "innerloop":self.cfg.general.innerloop,
                "partial":self.cfg.general.partial,
                "degree":round(res["degree"],6),
                "clustering":round(res["clustering"],6),
                "orbit":round(res["orbit"],6),
                "non_iso":round(res["sampling/frac_non_iso"],6),
                "uniq_frac":round(res["sampling/frac_unique"],6)
            }
            if self.cfg.dataset.name =="planar":
                write_dict["planar_acc"]= round(res["planar_acc"],6)
                # avgscore = res["planar_acc"]
            else:
                write_dict["sbm_acc"]= round(res["sbm_acc"],6)
                # avgscore = res["sbm_acc"]
            print("write to log file")
            line = json.dumps(write_dict)+"\n"
            logf.write(line)
            logf.close()
        # print("Computing sampling metrics...")
        # self.sampling_metrics.reset()
        # self.sampling_metrics(samples, self.name,
        #                       self.current_epoch, self.val_counter, test=True)
        # self.sampling_metrics.reset()
        # print("Done.")


    def test_epoch_end_ppo(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        wandb.log({"test/epoch_NLL": metrics[0],
                   "test/X_kl": metrics[1],
                   "test/E_kl": metrics[2],
                   "test/X_logp": metrics[3],
                   "test/E_logp": metrics[4]}, commit=False)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
              f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        print(f'Test loss: {test_nll :.4f}')

        # samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_generate = 30000
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        # start = time.time()
        result_list = []
        now = datetime.now()
        timestamp = now.timestamp()
        dt = datetime.fromtimestamp(timestamp)
        formatted_dt = dt.strftime("%m-%d-%H:%M:%S")
        filename = "generated_samples_ppo_{}_num{}.pt".format(formatted_dt,id)
        while samples_left_to_generate > 0:
            print(f'Samples left to generate: {samples_left_to_generate}/'
                  f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            result = self.sample_ppo(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps)
            filename = "generated_samples_ppo_{}_num{}.pt".format(formatted_dt,id)
            torch.save(result,filename)
            result_list.append(result)
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        now = datetime.now()
        timestamp = now.timestamp()
        dt = datetime.fromtimestamp(timestamp)
        formatted_dt = dt.strftime("%m-%d-%H:%M:%S")
        filename = "generated_samples_ppo_{}_num{}.pt".format(formatted_dt,id)
        torch.save(result_list,filename)

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities

        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None,
                                    :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(
            bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(
            input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(
            input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
            diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(
            noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(
            noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(
            prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(
            prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)
        # proby0 = (y.unsqueeze(1) @ Q0.y).squeeze(1)
        sampled0 = diffusion_utils.sample_discrete_features(
            probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        # print("y0 shape", y0.shape)
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        # print("recons y shape", sampled_0.y.shape,extra_data.y.shape)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
               ] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask,back_iter=None,seed=None):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        lowest_t = 0 if self.training else 1
        if back_iter is not None:
            t_int = back_iter*torch.ones(size=(X.size(0),1)).float().to(X.device)
        else:
            t_int = torch.randint(
                lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(
            t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(
            t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(
            t_normalized=t_float)      # (bs, 1)

        # (bs, dx_in, dx_out), (bs, de_in, de_out)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        # assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        # assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()
        # print(X.shape,Qtb.X.shape)
        # sys.exit()
        # Compute transition probabilities
        # print("check X, Q type",X.type(),Qtb.X.type())
        # print("shapes are {}/{}".format(X.shape,Qtb.X.shape))
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data
    @torch.no_grad()
    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False,no_prior=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(
            X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask)

        loss_term_0 = self.val_X_logp(
            X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        if no_prior:
            nlls = - log_pN + loss_all_t - loss_term_0
        else:
            nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(
            nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(
            nlls)        # Average over the batch

        wandb.log({"kl prior": kl_prior.mean(),
                   "Estimator loss terms": loss_all_t.mean(),
                   "log_pn": log_pN.mean(),
                   "loss_term_0": loss_term_0,
                   'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        # print("forward shae",X.shape,E.shape,y.shape)
        return self.model(X, E, y, node_mask)
    
    @torch.no_grad()
    def sample_ppo(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * \
                torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(
            0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        x, edge_index, edge_weight, batch = compress(X.cpu(),E.cpu(),node_mask.cpu())
        all_latent = [(x, edge_index, edge_weight, batch,y.cpu())]
        # print("latent shape is ",X.shape,E.shape,(X.sum(-1)==0).sum())
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
            x, edge_index, edge_weight, batch = compress(X.cpu(),E.cpu(),node_mask.cpu())
            all_latent.append((x, edge_index, edge_weight, batch, y.cpu()))
        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
        valid_list = compute_molecular_metrics_list(molecule_list,self.dataset_info)
        valid_list = torch.Tensor(valid_list)
        result = {"latents":all_latent,"molecules":molecule_list,"reward_list":valid_list,"node_mask":node_mask.cpu()}
        return result

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * \
                torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(
            0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T, "chain_step {}, T{}".format(number_chain_steps,self.T)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y


        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        return molecule_list
    @torch.no_grad()
    def sample_batch_seed(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, seed=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * \
                torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(
            0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y


        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        return molecule_list
    @torch.no_grad()
    def sample_batch_ppo(self, batch_size: int):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        self.model.eval()
        n_nodes = self.node_dist.sample_n(batch_size, self.device)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(
            0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        X_traj = []
        E_traj = []
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
        assert (E == torch.transpose(E, 1, 2)).all()
        X_traj.append(X.cpu())
        E_traj.append(E.cpu())
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
            X_traj.append(X.cpu())
            E_traj.append(E.cpu())
        #compute reward
        s0 = sampled_s.mask(node_mask, collapse=True)
        X, E, y = s0.X, s0.E, s0.y
        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
        if "ogbg" in self.cfg.dataset.name:
            valid_list,uniq,freq_list,reward_list = compute_molecular_metrics_list(molecule_list,self.dataset_info)
            for idx,freq in enumerate(freq_list):
                if freq>0:
                    reward_list[idx] = reward_list[idx]/freq
            validmean = np.array(valid_list).mean().item()
        elif self.cfg.dataset.name in ["zinc","moses"]:
            if self.train_fps is None:
                assert self.train_smiles is not None
                train_mols = [Chem.MolFromSmiles(smi) for smi in self.train_smiles]
                self.train_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in train_mols]
            smiles,valid_r,uniq,uniq_r,freq_list = gen_smile_list(molecule_list,self.dataset_info)
            valid_idx = [x for x in range(len(smiles))if smiles[x] is not None]
            valid_list = [x for x in smiles if x is not None]
            if len(valid_list)>0:
                if self.cfg.general.discrete:
                    if self.cfg.dataset.name == "moses":
                        valid_score_list = gen_score_disc_listmose(self.cfg.general.target_prop,valid_list,self.cfg.general.thres,self.train_fps)
                    else:
                        valid_score_list = gen_score_disc_list(self.cfg.general.target_prop,valid_list,self.cfg.general.thres,self.train_fps,self.cfg.general.weight_list)
                else:
                    valid_score_list = gen_score_list(self.cfg.general.target_prop,valid_list,self.train_fps)
            else:
                valid_score_list = []
            reward_list = [0]*len(smiles)
            # print("valid score list",valid_score_list)
            for idx,score in enumerate(valid_score_list):
                reward_list[valid_idx[idx]]=score
            reward_list = np.array(reward_list)
            validmean = (reward_list[reward_list!=-1]).mean().item()
            reward_list = reward_list.tolist()
            for idx,value in enumerate(freq_list):
                if value>0 and reward_list[idx]!=-1:
                    reward_list[idx] = reward_list[idx]/value
        elif self.cfg.dataset.name in ["planar","sbm"] and "nodes" not in self.cfg.dataset:
            if self.train_graphs is None:
                self.train_graphs = loader_to_nx(self.trainer.datamodule.train_dataloader())
            reward_list = graph_rewards(molecule_list,self.train_graphs,self.cfg.dataset.name)
            validmean = np.array(reward_list).mean().item()
        elif "nodes" in self.cfg.dataset:
            reward_list = toy_rewards(molecule_list)
            validmean = np.array(reward_list).mean().item()    
        else:
            print("unexpected datset option")
        advantages = torch.Tensor(reward_list)
        self.model.train()
        return torch.stack(X_traj),torch.stack(E_traj),node_mask,advantages,validmean
        
    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, times=1):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t,
                      'y_t': y_t, 't': t, 'node_mask': node_mask}

        extra_data = self.compute_extra_data(noisy_data)
        # print("saample y shape is", y_t.shape,extra_data.y.shape)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        # bs, n, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
        unnormalized_prob_X = weighted_X.sum(
            dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / \
            torch.sum(unnormalized_prob_X, dim=-1,
                      keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        # bs, N, d0, d_t-1
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / \
            torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        if times>1:
            sampled_s = []
            for _ in range(times):
                sampled_s.append(diffusion_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask))
            res = []
            for ss in sampled_s:
                X_s = F.one_hot(ss.X, num_classes=self.Xdim_output).float()
                E_s = F.one_hot(ss.E, num_classes=self.Edim_output).float()
                assert (E_s == torch.transpose(E_s, 1, 2)).all()
                assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)
                out_one_hot = utils.PlaceHolder(
                    X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
                out_discrete = utils.PlaceHolder(
                    X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
                res.append((out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)))
            return res
        
        sampled_s = diffusion_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask)
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(
            X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(
            X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def sample_p_zs_given_zt_ppo(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t,
                      'y_t': y_t, 't': t, 'node_mask': node_mask}

        extra_data = self.compute_extra_data(noisy_data)
        # print("saample y shape is", y_t.shape,extra_data.y.shape)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        # bs, n, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
        unnormalized_prob_X = weighted_X.sum(
            dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / \
            torch.sum(unnormalized_prob_X, dim=-1,
                      keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        # bs, N, d0, d_t-1
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / \
            torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()
        
        sampled_s = diffusion_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask)
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()
        
        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(
            X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        logpX,logpE = self.logp(prob_X,prob_E,None,out_one_hot.X,out_one_hot.E,None)
        return out_one_hot.mask(node_mask).type_as(y_t), (logpX,logpE)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)