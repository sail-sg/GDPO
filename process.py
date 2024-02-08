import os 
import torch

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
def changesize(data,batch_size,idx):
    x,edge_index,edge_attr,batch,y = data
    # print("changesize orig batch shape x shape",batch.shape,x.shape)
    dense_data, node_mask = to_dense(
        x, edge_index, edge_attr, batch)
    dense_data = dense_data.mask(node_mask)
    X, E = dense_data.X, dense_data.E
    B = X.shape[0]
    L,R = idx*batch_size,(idx+1)*batch_size
    X = X[L:R]
    E = E[L:R]
    new_mask = node_mask[L:R]
    y = y[L:R]
    # X = X.squeeze(2)
    E = E.squeeze(3)
    # print(X.shape,E.shape,"check graph shape")
    x, edge_index, edge_weight, batch = to_sparse_batch(X,E,new_mask)
    return x, edge_index, edge_weight, batch,y
def gennew(latents,rewards,batch_size,idx):
    new_latents = []
    for data in latents:
        new_latents.append(changesize(data,batch_size,idx))
    L,R = idx*batch_size,(idx+1)*batch_size
    new_rewards = rewards[L:R]
    return new_latents,new_rewards
def splitdata(root_dir,save_dir,batch_size):
    files = os.listdir(root_dir)
    ava = [x for x in files if "ppo" in x and "pt" in x]
    new_prefix = "ppo_split"
    orig_batch = 256
    splitnum = orig_batch//batch_size
    split_no = 0
    for name in ava:
        path = os.path.join(root_dir,name)
        data = torch.load(path)
        latents = data["latents"]
        rewards = data["reward_list"]
        for idx in range(splitnum):
            new_latents,new_rewards = gennew(latents,rewards,batch_size,idx)
            name = "_".join([new_prefix,str(split_no)])+".pt"
            savepth = os.path.join(save_dir,name)
            torch.save({"latents":new_latents,"reward_list":new_rewards},savepth)
            print("finished {}".format(savepth))
            split_no+=1
