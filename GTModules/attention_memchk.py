import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time
import math

from torch_scatter import scatter_add
from torch_sparse import SparseTensor
from torch_sparse import coalesce
from torch_sparse import spmm
from torch_sparse import transpose

def memchk(i, string=None, level=0, tic=None):

    toc = time.time()
    elapsed = round((toc - tic)*1000, 3)

    if string is None:
        print("{} : {} GB, {} ms".format(str(i) + ' '*level, round(torch.cuda.memory_allocated(0)/1024**3, 3), elapsed))
    else:
        print("{} : {} GB, {} ms, {}".format(str(i) + ' '*level, round(torch.cuda.memory_allocated(0)/1024**3, 3), elapsed, ' (' + string + ')'))

    i += 1

    return i, toc



class PositionalEncoding_carte_mlp(nn.Module):
    def __init__(self, out_channels):
        super(PositionalEncoding_carte_mlp, self).__init__()

        self.nn = nn.Sequential(nn.Linear(3, out_channels),
                                nn.ReLU(),
                                nn.Linear(out_channels, out_channels))

    def forward(self, x):

        # B, C, N, K
        x = x-x[:,:,:,0].unsqueeze(3) # B, C, N, K
        x = self.nn(x.permute(0, 2, 3, 1)) # B, 3, N, K -> B, N, K, 3 -> B, N, K, C

        return x.permute(0,3,1,2) # B, N, K, C -> B, C, N, K

def select_neighbors(x, idx):
    batch_size, num_dims, num_points = x.size()
    k = idx.size(-1)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points # 1, 1, 1
    idx = idx + idx_base # B, N, K
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous() # B, N, C
    x = x.view(batch_size*num_points, -1)[idx, :]
    x = x.view(batch_size, num_points, k, -1).permute(0, 3, 1, 2) # 1, 3, 1024, 20
    return x

def select_neighbors_nl(x, idx=None):
    # input
    # idx : B, G, 1, K'
    # x : B, C, N

    batch_size, num_dims, num_points = x.size()
    _, groups, _,k = idx.size()
    x = x.view(batch_size, -1, num_points)
    idx = idx.repeat(1,1,num_points,1) # B, G, 1, K' -> B, G, N, K'
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1, 1)*num_points # 1, 1, 1, 1
    idx = idx + idx_base # B, G, N, K
    _, num_dims, _ = x.size() # 3
    x = x.transpose(2, 1).unsqueeze(1).repeat(1,groups,1,1).contiguous() # B, G, N, C
    idx = idx.view(-1) # BGNK
    feature = x.view(batch_size*groups*num_points, -1)[idx, :] # 1024*8*20, C
    feature = feature.view(batch_size, groups, num_points, k, -1).permute(0, 1, 4, 2, 3) # B, G, N, K, C -> B, G, C, N, K
    return feature


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, ape=False, scale=False):
        super(AttentionConv, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.ape = ape
        self.scale = scale

        assert self.out_channels % self.groups == 0
        "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.out_channels = out_channels

        self.nl_ratio = 0.25
        self.nl_channels = int(self.nl_ratio*self.out_channels)
        self.l_channels = self.out_channels - self.nl_channels

        self.query_conv = nn.Conv2d(in_channels//2, self.l_channels, kernel_size=1, bias=bias)
        self.key_conv = nn.Conv2d(in_channels//2, self.l_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, self.l_channels, kernel_size=1, bias=bias)

        self.nl1_query_conv = nn.Conv2d(in_channels//2, self.nl_channels, kernel_size=1, bias=bias)
        self.nl1_key_conv = nn.Conv2d(in_channels//2, self.nl_channels, kernel_size=1, bias=bias)
        self.nl1_value_conv = nn.Conv2d(in_channels//2, self.nl_channels , kernel_size=1, bias=bias)

        self.nl2_value_conv = nn.Conv2d(in_channels//2, self.nl_channels , kernel_size=1, bias=bias)

        self.act = nn.Tanh()
        self.dp = nn.Dropout2d(p=0.2)

        if self.ape:
            self.local_shape = PositionalEncoding_carte_mlp(self.l_channels)
            self.nonlocal_shape = nn.Sequential(*[PositionalEncoding_carte_mlp(self.nl_channels//self.groups) for i in range(self.groups)])

    def forward(self, x, abs_x, idx, points):

        cnt = 0

        batch, channels, npoints, neighbors = x.size() # B, C, N, K

        # X : concat(x_j-x_i, x_i)
        # C + C' = C_out
        # idx : B, 1, N, K
        ''' 1. Local operation '''
        ''' 1.1. get point features '''
        #It was repeated for concatenate feature
        local_query = abs_x
        local_key = x[:, channels//2:, :, :] + x[:, :channels//2, :, :]
        local_value = x



        #MEMCHK# 0
        tic = time.time()
        cnt, tic = memchk(cnt, level=4, tic=tic, string="LocalOpStart")
        #MEMCHK#



        ''' 1.2. transform by Wq, Wk, Wv '''
        local_query = self.query_conv(local_query)  # B, C, N, K
        local_key = self.key_conv(local_key)        # B, C, N, K
        local_value = self.value_conv(local_value)  # B, C, N, K


        #MEMCHK# 1
        cnt, tic = memchk(cnt, level=4, tic=tic, string="LocalConv")
        #MEMCHK#



        ##Agument to one convolution is possible???
        ''' 1.3. Multi-heads for local operations. '''
        local_query = local_query.view(batch, self.groups, self.l_channels // self.groups, npoints, -1) # B, G, C//G, N, K
        local_key = local_key.view(batch, self.groups, self.l_channels // self.groups, npoints, -1) # B, G, C//G, N, K
        local_value = local_value.view(batch, self.groups, self.l_channels // self.groups, npoints, -1) # B, G, C//G, N, K




        #TODO RPE for non-local operation
        ''' 1.4. absolute positional encoding '''

        if self.ape:
            shape_encode = select_neighbors(points, idx.squeeze(1))
            shape_encode = self.local_shape(shape_encode) # B, 3, N, K -> B, C//G, N, K
            shape_encode = shape_encode.view(batch, self.groups, self.l_channels // self.groups, npoints, -1)
            local_key = local_key + shape_encode



        # k_out : B, C, N, K / self.rel_k : C, 1, K

        ''' 1.5. Addressing '''
        #TODO scaling coefficient
        if self.scale:
            scaler = torch.tensor([self.l_channels // self.groups]).cuda()
            attention = torch.rsqrt(scaler) * (local_query * local_key).sum(2) # B, G, N, K
        else:
            # Sum Channel-wise
            attention = (local_query * local_key).sum(2) # B, G, N, K

        attention = F.softmax(attention, dim=-1) # B, G, N, K

        ac_attention = attention

        attention = attention.unsqueeze(2).expand_as(local_value) # B, G, C//G, N, K
        local_feature = torch.einsum('bgcnk,bgcnk -> bgcn', attention, local_value)   #B, G, C//G, N

        local_feature = local_feature.contiguous().view(batch, -1, npoints, 1) # B, G, C//G, N -> B, C, N, 1

        #MEMCHK# 6
        cnt, tic = memchk(cnt, string="LocalOpEnd", level=4, tic=tic)
        #MEMCHK#

        #MEMCHK# 4
        cnt, tic = memchk(cnt, string="ACStart", level=4, tic=tic)
        #MEMCHK#


        ## Check out value after training

        ##Get Attention Centrality Value
        #idx_value, idx_score = self.sparse_attention_centrality(attention, idx)
        #idx_value, idx_score = self.loop_sparse_attention_centrality(attention, idx)
        #idx_value, idx_score = self.parallel_sparse_attention_centrality(ac_attention, idx)
        #idx_value, idx_score = self.attention_centrality(ac_attention, idx)
        idx_value, idx_score = self.scatter_sparse_attention_centrality(ac_attention, idx)

        #MEMCHK# 5
        cnt, tic = memchk(cnt, string="ACEnd", level=4, tic=tic)
        #MEMCHK#


        #MEMCHK# 5
        cnt, tic = memchk(cnt, string="NonLocalStart", level=4, tic=tic)
        #MEMCHK#

        ''' 1.7. Attention score for the node selection (Different per groups) '''
        ''' 1.8. Concat heads  '''
        ''' 2. Non-local MHA over selected nodes '''
        ''' - Memory friendly implementation '''

        ''' 2.1. transform by Wq, Wk, Wv & Multi-heads for non-local operation '''
        nonlocal_query = self.nl1_query_conv(abs_x)  #B, C, N, 1
        nonlocal_key = self.nl1_key_conv(abs_x)    #B, C, N, 1

        nonlocal_value_i = self.nl1_value_conv(abs_x)
        nonlocal_value_ij_i = self.nl2_value_conv(abs_x)


        #MEMCHK#
        cnt, tic = memchk(cnt, string="NonLocalConv", level=4, tic=tic)
        #MEMCHK#



        ''' 2.2. Multi-heads for non-local operations. '''
        nonlocal_query = nonlocal_query.view(batch, self.groups, self.nl_channels // self.groups, npoints)  # B, G, C//G, N
        nonlocal_key = nonlocal_key.view(batch, self.groups, self.nl_channels // self.groups, npoints)      # B, G, C//G, N
        nonlocal_value_i = nonlocal_value_i.view(batch, self.groups, self.nl_channels // self.groups, npoints)
        nonlocal_value_ij_i = nonlocal_value_ij_i.view(batch, self.groups, self.nl_channels // self.groups, npoints)




        ''' 2.3. select q2j, k2j, v2j by top-k idx '''
        idx_nl = idx_score
        idx_score = idx_score.repeat(1,1,self.nl_channels // self.groups, 1) # B, G, 1, K' -> B, G, C'//G, K'
        idx_value = idx_value.unsqueeze(3).repeat(1,1,self.nl_channels // self.groups, npoints,  1) # B, G, 1, K' -> B, G, C'//G, K'



        #q_nlj_out = torch.gather(q_nlj_out, 3, idx_score) # B, G, C'//G, N -> B, G, C'//G, K'
        nonlocal_key = torch.gather(nonlocal_key, 3, idx_score) # B, G, C'//G, N -> B, G, C'//G, K'
        nonlocal_value_ij_j = torch.gather(nonlocal_value_ij_i, 3, idx_score) # B, G, C'//G, N -> B, G, C'//G, K'



        ''' 2.4. expand 1i, 2i, 2j '''
        nonlocal_query = nonlocal_query.unsqueeze(4).repeat(1,1,1,1,neighbors) # B, G, C'//G, N, K
        nonlocal_key = nonlocal_key.unsqueeze(3).repeat(1,1,1,npoints,1)
        nonlocal_value_i = nonlocal_value_i.unsqueeze(4).repeat(1,1,1,1,neighbors)
        nonlocal_value_ij_i = nonlocal_value_ij_i.unsqueeze(4).repeat(1,1,1,1,neighbors)
        nonlocal_value_ij_j = nonlocal_value_ij_j.unsqueeze(3).repeat(1,1,1,npoints,1)


        ''' 2.5. aggregate all '''
        nonlocal_value = nonlocal_value_i - nonlocal_value_ij_i + nonlocal_value_ij_j
        nonlocal_value = nonlocal_value * self.act(idx_value)   # B, G, C'//G, N, K

        # points : B, 3, N
        # B, G, N, K

        if self.ape:
            selected_neighbors = select_neighbors_nl(points, idx=idx_nl) # B, 3, N -> B, G, 3, N, K
            shape_encode_nl = []
            for nl in range(self.groups):
                shape_encode_nl.append(self.nonlocal_shape[nl](selected_neighbors[:,nl,:,:,:]).unsqueeze(1))
            shape_encode_nl = torch.cat(shape_encode_nl, dim=1)
            nonlocal_key = nonlocal_key + shape_encode_nl


        ''' 2.7. Addressing '''
        if self.scale:
            scaler = torch.tensor([self.nl_channels / self.groups]).cuda()
            attention = torch.rsqrt(scaler) * (nonlocal_query * nonlocal_key).sum(2) # B, G, N, K
        else:
            attention = (nonlocal_query * nonlocal_key).sum(2) # B, G, N, K
        attention = F.softmax(attention, dim=-1) # B, G, N, K



        # dropout neighbors randomly
        attention = attention.permute(0, 2, 1, 3)
        attention = attention.contiguous().view(batch, npoints, self.groups*neighbors, 1) # B, N, G, K -> B, N, GK, 1
        attention = attention.permute(0, 2, 1, 3) # B, N, GK, 1 -> B, GK, N, 1
        attention = self.dp(attention).permute(0, 2, 1, 3) # B, GK, N, 1 -> B, N, GK, 1
        attention = attention.view(batch, npoints, self.groups, neighbors).permute(0, 2, 1, 3) # B, G, N, K -> B, N, G, K
        attention = attention.unsqueeze(2).repeat(1,1,self.nl_channels//self.groups,1,1) # B, G, C//G, N, K

        ''' 2.4. Scaling V '''
        nonlocal_feature = torch.einsum('bgcnk,bgcnk -> bgcn', attention, nonlocal_value)

        nonlocal_feature = nonlocal_feature.contiguous().view(batch, -1, npoints, 1) # B, G, C//G, N -> B, C, N, 1


        #MEMCHK#
        cnt, tic = memchk(cnt, level=4, tic=tic, string="NonlocalOpEnd")
        #MEMCHK#


        ''' 2.6. Concat '''
        feature = torch.cat([local_feature, nonlocal_feature], dim=1)


        return feature

    def attention_centrality(self, attention, idx):
        # legacy; O(N^2) implementation

        ''' 1.6. Attention score for the node selection (Different per groups) '''

        cnt = 0


        #MEMCHK#
        tic = time.time()
        cnt, tic = memchk(cnt, string="AC Start", level=8, tic=tic)
        #MEMCHK#


        batch, groups, npoints, neighbors = attention.size()
        idx_zeros = torch.zeros(batch, groups, npoints, npoints, device=attention.device)    # B, G, N, N


        #MEMCHK#
        cnt, tic = memchk(cnt, string="O(N^2)", level=8, tic=tic)
        #MEMCHK#


        idx_score = idx.repeat(1, groups, 1, 1)                                        # B, G, N, K
        idx_zeros.scatter_(dim=-1, index = idx_score, src=attention)

        # gradient path provided by out -> no linear projection layer like 'self-attention graph pooling' is needed.
        score = idx_zeros.sum(dim=2, keepdim=True) # B, G, 1, N
        # fullnl instance
        idx_value, idx_score = score.topk(k=neighbors, dim=3) # B, G, 1, N -> B, G, 1, K'


        #MEMCHK#
        cnt, tic = memchk(cnt, string="AC End", level=8, tic=tic)
        #MEMCHK#


        return idx_value, idx_score



    def scatter_sparse_attention_centrality(self, attention, idx):
        # O(N) implementation


        cnt = 0

        #MEMCHK#
        tic = time.time()
        cnt, tic = memchk(cnt, string="AC Start", level=8, tic=tic)
        #MEMCHK#


        batch, groups, npoints, neighbors = attention.size()
        idx = idx.repeat(1, groups, 1, 1).contiguous() # B, G, N, K
        centrality = torch.zeros(batch, groups, npoints, device='cuda')

        idx = idx.view(batch, groups, -1) # B, G, NK
        attention = attention.view(batch, groups, -1) # B, G, NK

        #MEMCHK#
        cnt, tic = memchk(cnt, string="ScatterStart", level=8, tic=tic)
        #MEMCHK#

        centrality = scatter_add(src=attention, index=idx, out=centrality, dim=2)

        #MEMCHK#
        cnt, tic = memchk(cnt, string="ScatterEnd", level=8, tic=tic)
        #MEMCHK#

        centrality = centrality.unsqueeze(2)

        idx_value, idx_score = centrality.topk(k=neighbors, dim=3) # B, G, 1, N -> B, G, 1, K'

        #MEMCHK#
        cnt, tic = memchk(cnt, string="AC End", level=8, tic=tic)
        #MEMCHK#
        
        return idx_value, idx_score


    def parallel_sparse_attention_centrality(self, attention, idx):
        # O(N) implementation


        cnt = 0

        #MEMCHK#
        tic = time.time()
        cnt, tic = memchk(cnt, string="AC Start", level=8, tic=tic)
        #MEMCHK#


        batch, groups, npoints, neighbors = attention.size()

        idx_arange = torch.tensor([[npoints*i]*npoints*neighbors for i in range(batch*groups)], device='cuda').flatten() # NK -> 11NK

        idx_self = torch.tensor([[i]*neighbors for i in range(npoints)], device='cuda').flatten().unsqueeze(0).unsqueeze(0) # NK -> 11NK
        idx_self = idx_self.repeat(batch, groups, 1) # BGNK
        mtrx = torch.tensor([[1.]*npoints*batch*groups], device='cuda').T

        idx_neighbor_flatten = idx.repeat(1, groups, 1, 1).flatten() # BGNK
        idx_neighbor_flatten = (idx_arange + idx_neighbor_flatten).unsqueeze(0) # 1, BGNK

        idx_self_flatten = idx_self.flatten() # BGNK
        idx_self_flatten = (idx_arange + idx_self_flatten).unsqueeze(0) # 1, BGNK

        index = torch.cat([idx_self_flatten, idx_neighbor_flatten], dim=0) # 2, BGNK

        attention_flatten = attention.flatten() # BGNK

        #MEMCHK#
        cnt, tic = memchk(cnt, string="CoalesceStart", level=8, tic=tic)
        #MEMCHK#

        index_s, value_s = coalesce(index, attention_flatten, npoints*batch*groups, npoints*batch*groups)
        index_t, value_t = transpose(index_s, value_s, npoints*batch*groups, npoints*batch*groups)

        out = spmm(index_t, value_t, npoints*batch*groups, npoints*batch*groups, mtrx)

        #MEMCHK#
        cnt, tic = memchk(cnt, string="CoalesceEnd", level=8, tic=tic)
        #MEMCHK#


        out = out.view(batch, groups, npoints, 1)
        score = out.permute(0, 1, 3, 2)
        idx_value, idx_score = score.topk(k=neighbors, dim=3) # B, G, 1, N -> B, G, 1, K'

        #MEMCHK#
        cnt, tic = memchk(cnt, string="AC End", level=8, tic=tic)
        #MEMCHK#


        
        return idx_value, idx_score



    def loop_sparse_attention_centrality(self, attention, idx):
        # O(N) implementation

        batch, groups, npoints, neighbors = attention.size()
        idx_tag = torch.tensor([[i]*neighbors for i in range(npoints)], device='cuda').flatten().unsqueeze(0)
        mtrx = torch.tensor([[1.]*npoints], device='cuda').T

        score = []

        for i in range(batch):
            idx_flatten = idx[i].flatten().unsqueeze(0) # NK
            index = torch.cat([idx_tag, idx_flatten], dim=0)
            score_group = []
            for j in range(groups):
                attention_flatten = attention[i][j].flatten()
                index_s, value_s = coalesce(index, attention_flatten, npoints, npoints)
                index_t, value_t = transpose(index_s, value_s, npoints, npoints)
                out = spmm(index_t, value_t, npoints, npoints, mtrx)
                score_group.append(out)
                if j == groups-1:
                    score.append(torch.cat(score_group, dim=1).unsqueeze(0))
            if i == batch-1:
                final_score = torch.cat(score, dim=0)

        # fullnl instance

        final_score = final_score.unsqueeze(3).permute(0, 2, 3, 1)
        idx_value, idx_score = final_score.topk(k=neighbors, dim=3) # B, G, 1, N -> B, G, 1, K'
        return idx_value, idx_score


