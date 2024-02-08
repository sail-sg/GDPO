# regular PyTorch
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Dataset
import os
import torch
import torch_geometric
# lightning 风格
import pytorch_lightning as pl
from torch_geometric.utils import to_dense_adj, to_dense_batch
def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E
class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self
def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    if len(E.shape)==3:
        E = E.unsqueeze(3)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask
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
        # print(adj.shape,mask.shape,"check shape")
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
class PPODataset(Dataset):
    def __init__(self, data_dir,f_list,batch_size,sample_rate):
        """ This class can be used to load the comm20, sbm and planar datasets. """
        self.data_list = [os.path.join(data_dir,x) for x in f_list]
        self.batch_size = batch_size
        self.full_trace = 500
        self.sample_rate = sample_rate
        self.sample_trace = int(self.sample_rate*self.full_trace)
    def __len__(self):

        return len(self.data_list)*self.sample_trace
    def __getitem__(self, idx):
        sample_pos = idx//self.sample_trace
        data_path = self.data_list[sample_pos]
        data = torch.load(data_path)
        latents = data["latents"]
        step = (idx%self.sample_trace)*int(1/self.sample_rate)
        now_data = latents[step]
        prev_data = latents[step+1]
        rewards = data["reward_list"]
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        resize_now = self.changesize(now_data)
        resize_prev = self.changesize(prev_data)
        advantages = advantages[:self.batch_size]
        return resize_now,resize_prev,step,advantages
    def changesize(self,data):
        x,edge_index,edge_attr,batch,y = data
        # print("changesize orig batch shape x shape",batch.shape,x.shape)
        dense_data, node_mask = to_dense(
            x, edge_index, edge_attr, batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        X = X[:self.batch_size]
        E = E[:self.batch_size]
        new_mask = node_mask[:self.batch_size]
        y = y[:self.batch_size]
        # X = X.squeeze(2)
        E = E.squeeze(3)
        # print(X.shape,E.shape,"check graph shape")
        x, edge_index, edge_weight, batch = to_sparse_batch(X,E,new_mask)
        return x, edge_index, edge_weight, batch,y

class PPOCLSDataset(Dataset):
    def __init__(self, data_dir,f_list,batch_size):
        """ This class can be used to load the comm20, sbm and planar datasets. """
        self.data_list = [os.path.join(data_dir,x) for x in f_list]
        self.batch_size = batch_size
    def __len__(self):

        return len(self.data_list)
    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        data = torch.load(data_path)
        latents = data["latents"]
        x0 = latents[-1]
        rewards = data["reward_list"]
        resize_now = self.changesize(x0)
        return resize_now,rewards
    def changesize(self,data):
        x,edge_index,edge_attr,batch,y = data
        # print("changesize orig batch shape x shape",batch.shape,x.shape)
        dense_data, node_mask = to_dense(
            x, edge_index, edge_attr, batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        X = X[:self.batch_size]
        E = E[:self.batch_size]
        new_mask = node_mask[:self.batch_size]
        y = y[:self.batch_size]
        # X = X.squeeze(2)
        E = E.squeeze(3)
        # print(X.shape,E.shape,"check graph shape")
        x, edge_index, edge_weight, batch = to_sparse_batch(X,E,new_mask)
        return x, edge_index, edge_weight, batch,y
import random
class PPODataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = None,total=400,batch_size=32,sample_rate=0.1,orig_datamodule=None):
        super().__init__()
        all_f = os.listdir(data_dir)
        self.data_dir = data_dir
        self.data_file = [x for x in all_f if "ppo" in x and ".pt" in x]
        random.shuffle(self.data_file)
        # print(len(self.data_file),type(self.data_file))
        self.data_file = self.data_file[:total]
        self.batch_size = batch_size
        self.total = total
        self.sample_rate = sample_rate
        self.orig_datamodule = orig_datamodule

    def prepare_data(self):
        # Assign train/val datasets for use in dataloaders
        self.train_data = PPODataset(self.data_dir,self.data_file[:int(1.*self.total)],self.batch_size,self.sample_rate)
        self.train_loader = DataLoader(self.train_data, batch_size=None,num_workers=16,shuffle=True)
        self.val_loader = self.orig_datamodule.val_dataloader()
        self.test_loader = self.orig_datamodule.test_dataloader()
    # 以下方法创建不同阶段的数据加载器
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

import random
class PPOCLSDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = None,total=400,batch_size=32):
        super().__init__()
        all_f = os.listdir(data_dir)
        self.data_dir = data_dir
        self.data_file = [x for x in all_f if "ppo" in x and ".pt" in x]
        random.shuffle(self.data_file)
        # print(len(self.data_file),type(self.data_file))
        self.data_file = self.data_file[:total]
        self.batch_size = batch_size
        self.total = total

    def prepare_data(self):
        # Assign train/val datasets for use in dataloaders
        self.train_data = PPOCLSDataset(self.data_dir,self.data_file[:int(0.8*self.total)],self.batch_size)
        self.train_loader = DataLoader(self.train_data, batch_size=None,num_workers=16,shuffle=True)
        self.val_data = PPOCLSDataset(self.data_dir,self.data_file[int(0.8*self.total):int(0.9*self.total)],self.batch_size)
        self.val_loader = DataLoader(self.train_data, batch_size=None,num_workers=16,shuffle=True)
        self.test_data = PPOCLSDataset(self.data_dir,self.data_file[int(0.9*self.total):int(1.*self.total)],self.batch_size)
        self.test_loader = DataLoader(self.train_data, batch_size=None,num_workers=16,shuffle=True)

    # 以下方法创建不同阶段的数据加载器
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

if __name__ == "__main__":
    datamodule = PPOCLSDataModule()
    datamodule.prepare_data()
    train_loader = datamodule.train_dataloader()
    for x in train_loader:
        print(len(x))
        now_data = x[0]
        x,edge_index,edge_attr,batch,y = now_data
        print(x.shape,batch.shape,"check output batch")
        dense_data, node_mask = to_dense(
            x, edge_index, edge_attr, batch)
        X, E = dense_data.X, dense_data.E
        print(X.shape,E.shape)
        break