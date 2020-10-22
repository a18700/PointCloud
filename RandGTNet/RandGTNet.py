import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
from helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix

import time
from torch_scatter import scatter_add

def memchk(i, string=None, level=0, tic=None):

    toc = time.time()
    elapsed = round((toc - tic)*1000, 3)

    if string is None:
        print("{} : {} GB, {} ms".format(str(i) + ' '*level, round(torch.cuda.memory_allocated(0)/1024**3, 3), elapsed))
    else:
        print("{} : {} GB, {} ms, {}".format(str(i) + ' '*level, round(torch.cuda.memory_allocated(0)/1024**3, 3), elapsed, ' (' + string + ')'))

    i += 1

    return i, toc



class Network(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.class_weights = self.config.class_weights

        if self.config.name == 'SemanticKITTI':
            self.fc0 = pt_utils.Conv1d(3, 8, kernel_size=1, bn=True)
        elif self.config.name == 'S3DIS':
            self.fc0 = pt_utils.Conv1d(6, 8, kernel_size=1, bn=True)
        elif self.config.name == 'Semantic3D':
            assert 1==2, "Not checked yet"


        self.GT_res_blocks = nn.ModuleList()

        d_in = 8

        # Encoder
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.GT_res_blocks.append(GT_res_block(d_in, d_out, self.config.groups))
            d_in = 2 * d_out

        d_out = d_in

        # First layer of Decoder
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)


        # Decoder
        self.decoder_blocks = nn.ModuleList()

        for j in range(self.config.num_layers):
            if self.config.name == "SemanticKITTI":
                if j < 3:
                    # j = 0, -j-2 = -2, 512 + 128*2 -> 256
                    # j = 1, -j-2 = -3, 256 + 64*2 -> 128
                    # j = 2, -j-2 = -4, 128 + 16*2 -> 32
                    d_in = d_out + 2 * self.config.d_out[-j-2]
                    d_out = 2 * self.config.d_out[-j-2]
                else:
                    # j = 3, 32 + 32 -> 32
                    d_in = 4 * self.config.d_out[-4]
                    d_out = 2 * self.config.d_out[-4]
            elif self.config.name in ["S3DIS", "Semantic3D"]:
                if j < 4:
                    d_in = d_out + 2 * self.config.d_out[-j-2]
                    d_out = 2 * self.config.d_out[-j-2]
                else:
                    d_in = 4 * self.config.d_out[-5]
                    d_out = 2 * self.config.d_out[-5]
                    
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))
            #print("j : {}, d_in : {}, d_out : {}".format(j, d_in, d_out))

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1,1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1,1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1,1), bn=False, activation=None)

    def forward(self, end_points):


        batch_size, num_points, _ = end_points['xyz'][0].shape
        idx_base = torch.arange(0, batch_size, device='cuda').view(-1, 1, 1)*num_points

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):

            #print("neigh idx : {}".format(end_points['neigh_idx'][i].shape))

            f_encoder_i = self.GT_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i], idx_base)

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i < 4:
                idx_base /= 4
            elif i == 4:
                idx_base /= 2
            else:
                assert 1==2, "Not implemented"

            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################


        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)

        end_points['logits'] = f_out


        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features



def compute_acc(end_points):

    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.cfg.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list



def gather_neighbour(pc, neighbor_idx, idx_base):
    batch_size = pc.shape[0]
    num_points = pc.shape[1]
    d = pc.shape[2]

    neighbor_idx = neighbor_idx + idx_base
    neighbor_idx = neighbor_idx.view(-1)

    pc = pc.contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = pc.view(batch_size*num_points, -1)[neighbor_idx, :]
    feature = feature.view(batch_size, num_points, -1, d) 
    return feature.permute(0,3,1,2).unsqueeze(1)


def batch_gather_neighbour(pc, neighbor_idx, idx_base, cq, cv):  
    # pc: B, G, N, 2*C+3
    # neighbor_idx: B, G, N, K

    # gather the coordinates or features of neighboring points
    batch_size = pc.shape[0]
    groups = pc.shape[1]
    num_points = pc.shape[2]

    features = []
    
    
    for g in range(groups):

        g_pc = pc[:,g,:,:] # B, N, C
        g_neighbor_idx = neighbor_idx[:,g,:,:] # B, N, K
        feature = gather_neighbour(g_pc, g_neighbor_idx, idx_base) # B, 1, C//G, N, K
        features.append(feature)

    features = torch.cat(features, dim=1) # B, G, 2*C//G+3, N, K
    
    key = features[:,:,:cq,:,:]
    value = features[:,:,cq:cq+cv,:,:]
    xyz = features[:,:,-3:,:,:]

    #cnt, #tic = memchk(cnt, level=12, tic=tic0, string="Batch Gather Neighbour End")

    return key, value, xyz

def relative_pos_encoding(xyz, f_xyz):
    #xyz ; bgnc
    #f_xyz ; bg(3 or c//g)nk

    k = f_xyz.shape[-1]

    neighbor_xyz = f_xyz.permute(0,1,3,4,2) 
    xyz_tile = xyz.unsqueeze(4).repeat(1, 1, 1, 1, k).permute(0,1,2,4,3)  # bgnck -> b*g*n*k*c

    #print(xyz.shape) # bgcnk
    #print(neighbor_xyz.shape) # bgnkc
    #print(xyz_tile.shape) # 

    relative_xyz = xyz_tile - neighbor_xyz  # b*g*n*k*c
    relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True)) # b*g*n*k*1
    relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # b*g*n*k*10

    return relative_feature # b*g*n*k*10


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz) # batch*channel*npoint*nsamples

        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature



class GT_res_block(nn.Module):
    def __init__(self, d_in, d_out, groups):
        super().__init__()

        # query/key for Local Block
        self.qk1 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=False)
        self.v1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)

        # query/key for Local/Global Blocks (i-1)
        self.qk2 = pt_utils.Conv2d(d_out, d_out, kernel_size=(1,1), bn=False)
        self.v2 = pt_utils.Conv2d(d_out, d_out//2, kernel_size=(1,1), bn=True)

        # value for Local/Global Blocks
        self.lblock = GTLModule(d_out, groups)
        self.gblock = GTModule(d_out, groups)

        # After Aggregation
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx, idx_base):
        q_pc = self.qk1(feature)  # Batch*channel*npoints*1
        f_pc = self.v1(feature)  # Batch*channel*npoints*1
        f_pc, f_xyz, attention_centrality = self.lblock(xyz, q_pc, f_pc, neigh_idx, idx_base)

        q_pc = self.qk2(f_pc)
        f_pc = self.v2(f_pc)
        # if more than 1 GTModule appended, use recurrence.
        f_pc, _ = self.gblock(xyz, f_xyz, q_pc, f_pc, neigh_idx, idx_base, attention_centrality)  # Batch*d_out*npoints*1
        
        f_pc = self.mlp2(f_pc)

        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class GTLModule(nn.Module):
    def __init__(self, d_out, groups):  #  d_in = d_out//2
        super().__init__()
        self.groups = groups

        self.lpe = pt_utils.Conv2d(10, d_out//(2*self.groups), kernel_size=(1,1), bn=True)

    def forward(self, xyz, query, value, neigh_idx, idx_base):  # feature: Batch*channel*npoints*1

        b, cv, n, _ = value.shape
        cq = query.shape[1]
        k = neigh_idx.shape[-1]
        
        '''
        print("b : {}".format(b))
        print("cv : {}".format(cv))
        print("n : {}".format(n))
        print("cq : {}".format(cq))
        print("k : {}".format(k))
        '''

        # idx
        neigh_idx = neigh_idx.contiguous().unsqueeze(1).repeat(1, self.groups, 1, 1) #b,g,n,k

        # Multi-head query
        xyz = xyz.contiguous().unsqueeze(1).repeat(1, self.groups, 1, 1)
        lq = query.contiguous().view(b, self.groups, cq//self.groups, n)
        lk = query.contiguous().view(b, self.groups, cq//self.groups, n)
        lv = value.contiguous().view(b, self.groups, cv//self.groups, n)

        # Local Gathering and Concat PE
        temp = torch.cat([lk, lv, xyz.permute(0,1,3,2)], dim=2).permute(0,1,3,2) # b*g*(catted c)*n -> b,g,n,c

        lk, lv, f_xyz = batch_gather_neighbour(temp, neigh_idx, idx_base, cq//self.groups, cv//self.groups)  # b,g,c//g,n,k

        # Local PE
        f_xyz = relative_pos_encoding(xyz, f_xyz)  # batch*npoint*nsamples*10 / b*g*10*n*k
        f_xyz = f_xyz.permute(0, 4, 2, 1, 3) # b, c//g or 10, n, g, k

        f_xyz = f_xyz.contiguous().view(b, f_xyz.shape[1], n, self.groups*k) # b, c//g, n, gk
        f_xyz = self.lpe(f_xyz) # b,g,c//g,n,k, batch*channel*npoint*nsamples
        f_xyz = f_xyz.view(b, cv//self.groups, n, self.groups, -1)
        f_xyz = f_xyz.permute(0, 3, 1, 2, 4) # b*g*c//g*n*k

        # Get Attention
        la = (lq.unsqueeze(-1) * lk).sum(2) # b*g*n*k
        la = F.softmax(la, dim=-1) #b*g*n*k
        
        lv = torch.cat([lv, f_xyz], dim=2) # b*g*(concated C)*n*k 
        lv = (la.unsqueeze(2) * lv).sum(-1) # b*g*c*n
        
        lv = lv.view(b, -1, n, 1)

        # Local Aggregation
        attention_centrality = sparse_attention_centrality(la, neigh_idx)  # Batch*channel*npoints*1

        return lv, f_xyz, attention_centrality




class GTModule(nn.Module):
    def __init__(self, d_out, groups):  #  d_in = d_out//2
        super().__init__()

        self.groups = groups

        self.gcv = d_out // (2 * 2) # first 2 for halfed feature, second 4 for local/non-local ratio
        self.lcv = d_out // 2 - self.gcv

        self.lpe = pt_utils.Conv2d(d_out//(2*self.groups), self.lcv//(self.groups), kernel_size=(1, 1), bn=True)
        self.gpe = pt_utils.Conv2d(10, self.gcv//(self.groups), kernel_size=(1,1), bn=True)
        

    def forward(self, xyz, l_xyz, query, value, neigh_idx, idx_base, attention_centrality):  # feature: Batch*channel*npoints*1

        # Total Feature
        # : gcv + lcv + gcv(pos) + lcv(pos)

        b, cv, n, _ = value.shape
        cq = query.shape[1]
        k = neigh_idx.shape[-1]

        gcv = self.gcv 
        lcv = self.lcv 

        gcq = cq//2
        lcq = cq - gcq

        '''
        print("b : {}".format(b))
        print("cv : {}".format(cv))
        print("n : {}".format(n))
        print("cq : {}".format(cq))
        print("k : {}".format(k))

        print("gcv : {}".format(gcv))
        print("lcv : {}".format(lcv))
        print("gcq : {}".format(gcq))
        print("lcq : {}".format(lcq))

        print("groups : {}".format(self.groups))
        '''
        # idxes
        neigh_idx = neigh_idx.contiguous().unsqueeze(1).repeat(1, self.groups, 1, 1) #b,g,n,k
        _, ac_idx = attention_centrality.topk(k=k, dim=2) # b*g*k
        ac_idx = ac_idx.unsqueeze(2).repeat(1,1,n,1) # b*g*n*k

        # Multi-head query
        xyz = xyz.contiguous().unsqueeze(1).repeat(1, self.groups, 1, 1)
        lq = query[:,:lcq,:,:].contiguous().view(b, self.groups, lcq//self.groups, n)
        lk = query[:,:lcq,:,:].contiguous().view(b, self.groups, lcq//self.groups, n)
        lv = value[:,:lcv,:,:].contiguous().view(b, self.groups, lcv//self.groups, n)

        gq = query[:,lcq:,:,:].contiguous().view(b, self.groups, gcq//self.groups, n)
        gk = query[:,lcq:,:,:].contiguous().view(b, self.groups, gcq//self.groups, n)
        gv = value[:,lcv:,:,:].contiguous().view(b, self.groups, gcv//self.groups, n)

        # Local Gathering and Concat PE
        ltemp = torch.cat([lk, lv, xyz.permute(0,1,3,2)], dim=2).permute(0,1,3,2) # b*g*(catted c)*n -> b*g*n*c
        gtemp = torch.cat([gk, gv, xyz.permute(0,1,3,2)], dim=2).permute(0,1,3,2) # b*g*(catted c)*n -> b*g*n*c

        lk, lv, _ = batch_gather_neighbour(ltemp, neigh_idx, idx_base, lcq//self.groups, lcv//self.groups)  # b,g,c//g,n,k
        gk, gv, g_xyz = batch_gather_neighbour(gtemp, ac_idx, idx_base, gcq//self.groups, gcv//self.groups) 

        # Local PE
        l_xyz = l_xyz.permute(0, 2, 3, 1, 4) # b, c//g, n, g, k
        l_xyz = l_xyz.view(b, cv//(self.groups), n, self.groups*k) # b, c//g, n, gk
        l_xyz = self.lpe(l_xyz) # b,g,c//g,n,k, batch*channel*npoint*nsamples
        l_xyz = l_xyz.view(b, lcv//self.groups, n, self.groups, k)
        l_xyz = l_xyz.permute(0, 3, 1, 2, 4) # b*g*c//g*n*k

        # Global PE
        g_xyz = relative_pos_encoding(xyz, g_xyz)  # batch*npoint*nsamples*10 / b*g*10*n*k
        g_xyz = g_xyz.permute(0, 4, 2, 1, 3) # b, c//g, n, g, k
        g_xyz = g_xyz.contiguous().view(b, g_xyz.shape[1], n, self.groups*k) # b, c//g, n, gk
        g_xyz = self.gpe(g_xyz) # b,g,c//g,n,k,   batch*channel*npoint*nsamples
        
        g_xyz = g_xyz.view(b, gcv//self.groups, n, self.groups, -1)
        g_xyz = g_xyz.permute(0, 3, 1, 2, 4) # b*g*c//g*n*k


        # Get Attention
        la = (lq.unsqueeze(-1) * lk).sum(2) # b*g*n*k
        la = F.softmax(la, dim=-1) #b*g*n*k
        
        lv = torch.cat([lv, l_xyz], dim=2) # b*g*(concated C)*n*k 
        lv = (la.unsqueeze(2) * lv).sum(-1) # b*g*c*n
        
        ga = (gq.unsqueeze(-1) * gk).sum(2) # b*g*n*k
        ga = F.softmax(ga, dim=-1) #b*g*n*k
        
        gv = torch.cat([gv, g_xyz], dim=2) # b*g*(concated C)*n*k 
        gv = (ga.unsqueeze(2) * gv).sum(-1) # b*g*c*n
        
        v = torch.cat([lv, gv], dim=2)
        v = v.view(b, -1, n, 1)
        
        # Local Aggregation
        attention_centrality = sparse_attention_centrality(la, neigh_idx)  # Batch*channel*npoints*1

        return v, attention_centrality




class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


class GT_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set, neigh_idx):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3) # bcnk
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)

        attention_centrality = sparse_attention_centrality(att_scores, neigh_idx)

        return f_agg


def sparse_attention_centrality(attention, idx, groups = 1):
    # O(N) implementation


    batch, groups, npoints, neighbors = attention.size()
    centrality = torch.zeros(batch, groups, npoints, device='cuda')

    idx = idx.view(batch, groups, -1) # B, G, NK
    attention = attention.view(batch, groups, -1) # B, G, NK

    centrality = scatter_add(src=attention, index=idx, out=centrality, dim=2) # B, G, N
    #centrality = centrality.unsqueeze(2)

    #idx_value, idx = centrality.topk(k=neighbors, dim=2) # B, G, N -> B, G, K'

    return centrality



def compute_loss(end_points, cfg):

    logits = end_points['logits']
    labels = end_points['labels']
  

    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
    labels = labels.reshape(-1)
    

    # Boolean mask of points that should be ignored
    ignored_bool = labels == 0
    for ign_label in cfg.ignored_label_inds:
        ignored_bool = ignored_bool | (labels == ign_label)

    # Collect logits and labels that are not ignored
    
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]
    
    

    # Reduce label values in the range of logit shape
    reducing_list = torch.range(0, cfg.num_classes).long().cuda()
    inserted_value = torch.zeros((1,)).long().cuda()
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    loss = get_loss(valid_logits, valid_labels, cfg.class_weights)
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
    end_points['loss'] = loss
    return loss, end_points


def get_loss(logits, labels, pre_cal_weights):
    # calculate the weighted cross entropy according to the inverse frequency
    class_weights = torch.from_numpy(pre_cal_weights).float().cuda()
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss
