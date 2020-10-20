import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
from helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix

from PPNet_ops.pt_custom_ops.pt_utils import *
from torch_scatter import scatter_add
import time



def scatter_sparse_attention_centrality(attention, idx):
    # O(N) implementation
    batch, groups, npoints, neighbors = attention.size()
    idx = idx.repeat(1, groups, 1, 1).contiguous() # B, G, N, K
    centrality = torch.zeros(batch, groups, npoints, device='cuda')

    idx = idx.view(batch, groups, -1) # B, G, NK
    attention = attention.view(batch, groups, -1) # B, G, NK

    centrality = scatter_add(src=attention, index=idx, out=centrality, dim=2) # B, G, N
    #centrality = centrality.unsqueeze(2)

    #idx_value, idx = centrality.topk(k=neighbors, dim=2) # B, G, N -> B, G, K'

    return centrality



def build_scene_segmentation(config):
    model = SceneSegmentationModel(config, config.backbone, config.head, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio)
    return model


class SceneSegmentationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, input_features_dim, radius, sampleDl, nsamples, npoints, width=144, depth=2, bottleneck_ratio=2):
        super(SceneSegmentationModel, self).__init__()
        if backbone == 'resnet':
            if config.local_aggregation_type == 'gt':
                self.backbone = GTResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints, width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
            else:
                self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints, width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)


        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Multi-Part Segmentation Model")

        if head == 'resnet_scene_seg':
            self.segmentation_head = SceneSegHeadResNet(num_classes, width, radius, nsamples)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Multi-Part Segmentation Model")

    def forward(self, end_points):
        end_points = self.backbone(end_points)
        return self.segmentation_head(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class MultiInputSequential(nn.Sequential):

    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input



class LocalAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """LocalAggregation operators

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(LocalAggregation, self).__init__()
        if config.local_aggregation_type == 'pospool':
            self.local_aggregation_operator = PosPool(in_channels, out_channels, radius, nsample, config)
        elif config.local_aggregation_type == 'adaptive_weight':
            self.local_aggregation_operator = AdaptiveWeight(in_channels, out_channels, radius, nsample, config)
        elif config.local_aggregation_type == 'pointwisemlp':
            self.local_aggregation_operator = PointWiseMLP(in_channels, out_channels, radius, nsample, config)
        elif config.local_aggregation_type == 'pseudo_grid':
            self.local_aggregation_operator = PseudoGrid(in_channels, out_channels, radius, nsample, config)
        elif config.local_aggregation_type == 'gt':
            self.local_aggregation_operator = PosPool(in_channels, out_channels, radius, nsample, config)

        else:
            raise NotImplementedError(f'LocalAggregation {config.local_aggregation_type} not implemented')


    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        return self.local_aggregation_operator(query_xyz, support_xyz, query_mask, support_mask, support_features)




class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio, radius, nsample, config,
                 downsample=False, sampleDl=None, npoint=None, attention_centrality=None):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        
        if downsample:
            self.maxpool = MaskedMaxPool(npoint, radius, nsample, sampleDl)

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels // bottleneck_ratio, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channels // bottleneck_ratio, momentum=config.bn_momentum),
                                   nn.ReLU(inplace=True))

        self.local_aggregation = LocalAggregation(out_channels // bottleneck_ratio,
                                                  out_channels // bottleneck_ratio,
                                                  radius, nsample, config)
        self.conv2 = nn.Sequential(nn.Conv1d(out_channels // bottleneck_ratio, out_channels, kernel_size=1, bias=False), nn.BatchNorm1d(out_channels, momentum=config.bn_momentum))

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum))


    def forward(self, xyz, mask, features):
        if self.downsample:
            sub_xyz, sub_mask, sub_features = self.maxpool(xyz, mask, features)
            query_xyz = sub_xyz
            query_mask = sub_mask
            identity = sub_features

        else:
            query_xyz = xyz
            query_mask = mask
            identity = features

        output = self.conv1(features)
        output = self.local_aggregation(query_xyz, xyz, query_mask, mask, output)
        output = self.conv2(output)

        if self.in_channels != self.out_channels:
            identity = self.shortcut(identity)

        output += identity
        output = self.relu(output)

        return query_xyz, query_mask, output






class ResNet(nn.Module):
    def __init__(self, config, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        """Resnet Backbone

        Args:
            config: config file.
            input_features_dim: dimension for input feature.
            radius: the base ball query radius.
            sampleDl: the base grid length for sub-sampling.
            nsamples: neighborhood limits for each layer, a List of int.
            npoints: number of points after each sub-sampling, a list of int.
            width: the base channel num.
            depth: number of bottlenecks in one stage.
            bottleneck_ratio: bottleneck ratio.

        Returns:
            A dict of points, masks, features for each layer.
        """
        super(ResNet, self).__init__()

        self.input_features_dim = input_features_dim

        self.conv1 = nn.Sequential(nn.Conv1d(input_features_dim, width // 2, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(width // 2, momentum=config.bn_momentum),
                                   nn.ReLU(inplace=True))


        self.la1 = LocalAggregation(width // 2, width // 2, radius, nsamples[0], config)
        self.btnk1 = Bottleneck(width // 2, width, bottleneck_ratio, radius, nsamples[0], config)

        self.layer1 = MultiInputSequential()
        sampleDl *= 2
        self.layer1.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[0], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[0]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer1.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[1], config))

        self.layer2 = MultiInputSequential()
        sampleDl *= 2
        self.layer2.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[1], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[1]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer2.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[2], config))

        self.layer3 = MultiInputSequential()
        sampleDl *= 2
        self.layer3.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[2], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[2]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer3.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[3], config))

        self.layer4 = MultiInputSequential()
        sampleDl *= 2
        self.layer4.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[3], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[3]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer4.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[4], config))

    def forward(self, end_points):
        """
        Args:
            xyz: (B, N, 3), point coordinates
            mask: (B, N), 0/1 mask to distinguish padding points1
            features: (B, 3, input_features_dim), input points features.
            end_points: a dict

        Returns:
            end_points: a dict contains all outputs
        """
        if not end_points: end_points = {}

        # res1
        features = self.conv1(end_points['features'])

        features = self.la1(end_points['xyz'][0], end_points['xyz'][0], end_points['mask'], end_points['mask'], features, end_points['neigh_idx'][0])
        xyz, mask, features = self.btnk1(end_points['xyz'][0], end_points['mask'], features, end_points['neigh_idx'][0])
        end_points['res1_xyz'] = xyz
        end_points['res1_mask'] = mask
        end_points['res1_features'] = features

        # res2
        xyz, mask, features = self.layer1(xyz, mask, features, end_points['neigh_idx'][1])
        end_points['res2_xyz'] = xyz
        end_points['res2_mask'] = mask
        end_points['res2_features'] = features

        # res3
        xyz, mask, features = self.layer2(xyz, mask, features, end_points['neigh_idx'][2])
        end_points['res3_xyz'] = xyz
        end_points['res3_mask'] = mask
        end_points['res3_features'] = features

        # res4
        xyz, mask, features = self.layer3(xyz, mask, features, end_points['neigh_idx'][3])
        end_points['res4_xyz'] = xyz
        end_points['res4_mask'] = mask
        end_points['res4_features'] = features

        # res5
        xyz, mask, features = self.layer4(xyz, mask, features, end_points['neigh_idx'][4])
        end_points['res5_xyz'] = xyz
        end_points['res5_mask'] = mask
        end_points['res5_features'] = features

        return end_points



class GTResNet(nn.Module):
    def __init__(self, config, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        """Resnet Backbone

        Args:
            config: config file.
            input_features_dim: dimension for input feature.
            radius: the base ball query radius.
            sampleDl: the base grid length for sub-sampling.
            nsamples: neighborhood limits for each layer, a List of int.
            npoints: number of points after each sub-sampling, a list of int.
            width: the base channel num.
            depth: number of bottlenecks in one stage.
            bottleneck_ratio: bottleneck ratio.

        Returns:
            A dict of points, masks, features for each layer.
        """
        super(GTResNet, self).__init__()

        self.input_features_dim = input_features_dim

        self.conv1_queryandkey = nn.Conv1d(input_features_dim, width//2, kernel_size=1, bias=False)
        self.conv1_value = nn.Conv1d(input_features_dim, width//2, kernel_size=1, bias=False)
        self.conv1_bn = nn.BatchNorm1d(width // 2, momentum=config.bn_momentum),
        self.conv1_relu = nn.ReLU(inplace=True)

        self.a1 = GTLModuleV1(width // 2, width // 2, radius, nsamples[0], config)
        self.btnk1 = GTBottleneck(width // 2, width, bottleneck_ratio, radius, nsamples[0], config)

        self.layer1 = MultiInputSequential()
        sampleDl *= 2
        self.layer1.add_module("strided_bottleneck",
                               GTBottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[0], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[0]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer1.add_module(f"bottlneck{i}",
                                   GTBottleneck(width, width, bottleneck_ratio, radius, nsamples[1], config))

        self.layer2 = MultiInputSequential()
        sampleDl *= 2
        self.layer2.add_module("strided_bottleneck",
                               GTBottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[1], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[1]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer2.add_module(f"bottlneck{i}",
                                   GTBottleneck(width, width, bottleneck_ratio, radius, nsamples[2], config))

        self.layer3 = MultiInputSequential()
        sampleDl *= 2
        self.layer3.add_module("strided_bottleneck",
                               GTBottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[2], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[2]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer3.add_module(f"bottlneck{i}",
                                   GTBottleneck(width, width, bottleneck_ratio, radius, nsamples[3], config))
        
        self.layer4 = MultiInputSequential()
        sampleDl *= 2
        self.layer4.add_module("strided_bottleneck",
                               GTBottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[3], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[3]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer4.add_module(f"bottlneck{i}",
                                   GTBottleneck(width, width, bottleneck_ratio, radius, nsamples[4], config))

    def forward(self, end_points):
        """
        Args:
            xyz: (B, N, 3), point coordinates
            mask: (B, N), 0/1 mask to distinguish padding points1
            features: (B, 3, input_features_dim), input points features.
            end_points: a dict

        Returns:
            end_points: a dict contains all outputs
        """
        if not end_points: end_points = {}


        # res1
        #print("Res1")

        features_queryandkey = self.conv1_queryandkey(end_points['features'])
        features_value = self.conv1_value(end_points['features'])
        features, attention_centrality = self.a1(end_points['xyz'][0], end_points['xyz'][0], end_points['mask'], end_points['mask'], features_queryandkey, features_value, idx_knn=end_points['neigh_idx'][0])
        xyz, mask, features, attention_centrality, _ = self.btnk1(end_points['xyz'][0], end_points['mask'], features, attention_centrality, end_points['neigh_idx'][0])
        end_points['res1_xyz'] = xyz
        end_points['res1_mask'] = mask
        end_points['res1_features'] = features

         
        # res2
        #print("Res2")
        xyz, mask, features, attention_centrality, _ = self.layer1(xyz, mask, features, None, end_points['neigh_idx'][1])
        end_points['res2_xyz'] = xyz
        end_points['res2_mask'] = mask
        end_points['res2_features'] = features

        # res3
        #print("Res3")
        xyz, mask, features, attention_centrality, _ = self.layer2(xyz, mask, features, None, end_points['neigh_idx'][2])
        end_points['res3_xyz'] = xyz
        end_points['res3_mask'] = mask
        end_points['res3_features'] = features


        # res4
        #print("Res4")
        xyz, mask, features, attention_centrality, _ = self.layer3(xyz, mask, features, None, end_points['neigh_idx'][3])
        end_points['res4_xyz'] = xyz
        end_points['res4_mask'] = mask
        end_points['res4_features'] = features


        # res5
        #print("Res5")
        xyz, mask, features, attention_centrality, _ = self.layer4(xyz, mask, features, None, end_points['neigh_idx'][4])
        end_points['res5_xyz'] = xyz
        end_points['res5_mask'] = mask
        end_points['res5_features'] = features

        return end_points


class SceneSegHeadResNet(nn.Module):
    def __init__(self, num_classes, width, base_radius, nsamples):
        """A scene segmentation head for ResNet backbone.

        Args:
            num_classes: class num.
            width: the base channel num.
            base_radius: the base ball query radius.
            nsamples: neighborhood limits for each layer, a List of int.

        Returns:
            logits: (B, num_classes, N)
        """
        super(SceneSegHeadResNet, self).__init__()
        self.num_classes = num_classes
        self.base_radius = base_radius
        self.nsamples = nsamples
        self.up0 = MaskedUpsample(radius=8 * base_radius, nsample=nsamples[3], mode='nearest')
        self.up1 = MaskedUpsample(radius=4 * base_radius, nsample=nsamples[2], mode='nearest')
        self.up2 = MaskedUpsample(radius=2 * base_radius, nsample=nsamples[1], mode='nearest')
        self.up3 = MaskedUpsample(radius=base_radius, nsample=nsamples[0], mode='nearest')

        self.up_conv0 = nn.Sequential(nn.Conv1d(24 * width, 4 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(4 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv1 = nn.Sequential(nn.Conv1d(8 * width, 2 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(2 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv2 = nn.Sequential(nn.Conv1d(4 * width, width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width),
                                      nn.ReLU(inplace=True))
        self.up_conv3 = nn.Sequential(nn.Conv1d(2 * width, width // 2, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width // 2),
                                      nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv1d(width // 2, width // 2, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(width // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(width // 2, num_classes, kernel_size=1, bias=True))

    def forward(self, end_points):
        features = self.up0(end_points['res4_xyz'], end_points['res5_xyz'],
                            end_points['res4_mask'], end_points['res5_mask'], end_points['res5_features'])
        features = torch.cat([features, end_points['res4_features']], 1)
        features = self.up_conv0(features)

        features = self.up1(end_points['res3_xyz'], end_points['res4_xyz'],
                            end_points['res3_mask'], end_points['res4_mask'], features)
        features = torch.cat([features, end_points['res3_features']], 1)
        features = self.up_conv1(features)

        features = self.up2(end_points['res2_xyz'], end_points['res3_xyz'],
                            end_points['res2_mask'], end_points['res3_mask'], features)
        features = torch.cat([features, end_points['res2_features']], 1)
        features = self.up_conv2(features)

        features = self.up3(end_points['res1_xyz'], end_points['res2_xyz'],
                            end_points['res1_mask'], end_points['res2_mask'], features)
        features = torch.cat([features, end_points['res1_features']], 1)
        features = self.up_conv3(features)

        end_points['logits'] = self.head(features)

        return end_points



# ATT AGG  ==============================================================================
class GTBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio, radius, nsample, config,
                 downsample=False, sampleDl=None, npoint=None, attention_centrality=None):
        super(GTBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        
        if downsample:
            self.maxpool = MaskedMaxPool(npoint, radius, nsample, sampleDl)

        self.query = nn.Conv1d(in_channels, out_channels // bottleneck_ratio, kernel_size=1, bias=False)
        self.value = nn.Conv1d(in_channels, out_channels // bottleneck_ratio, kernel_size=1, bias=False) 
        self.conv1_bn = nn.BatchNorm1d(out_channels // bottleneck_ratio, momentum=config.bn_momentum)
        self.conv1_relu = nn.ReLU(inplace=True)

        if downsample:
            self.aggregation = GTLModuleV1(out_channels // bottleneck_ratio,
                                                  out_channels // bottleneck_ratio,
                                                  radius, nsample, config)
        else:
            self.aggregation = GTModuleV1(out_channels // bottleneck_ratio,
                                                  out_channels // bottleneck_ratio,
                                                  radius, nsample, config)
        self.conv2 = nn.Sequential(nn.Conv1d(out_channels // bottleneck_ratio, out_channels, kernel_size=1, bias=False), nn.BatchNorm1d(out_channels, momentum=config.bn_momentum))

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum))


    def forward(self, xyz, mask, features, attention_centrality=None, idx_knn=None):

        if self.downsample:
            sub_xyz, sub_mask, sub_features = self.maxpool(xyz, mask, features)
            query_xyz = sub_xyz
            query_mask = sub_mask
            identity = sub_features

            queryandkey = self.query(sub_features) # B, C, N
            value = self.value(sub_features) # B, C, N

        else:
            query_xyz = xyz
            query_mask = mask
            identity = features
            
            queryandkey = self.query(features) # B, C, N
            value = self.value(features) # B, C, N

        output, attention_centrality = self.aggregation(query_xyz, query_xyz, query_mask, query_mask, queryandkey, value, attention_centrality, idx_knn)
        output = self.conv1_bn(output)
        output = self.conv1_relu(output)

        output = self.conv2(output)

        if self.in_channels != self.out_channels:
            identity = self.shortcut(identity)

        output += identity
        output = self.relu(output)

        return query_xyz, query_mask, output, attention_centrality, idx_knn


class GTLModuleV1(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A GT operator for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(GTLModuleV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.radius = radius. No Need for GTModule
        self.nsample = nsample
        self.position_embedding = config.position_embedding
        self.reduction = config.reduction
        self.output_conv = config.output_conv or (self.in_channels != self.out_channels)

        self.use_xyz = False

        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))


    def forward(self, query_xyz, support_xyz, query_mask, support_mask, queryandkey, value, attention_centrality=None, idx_knn=None):
        """
        Args:
           Query : (B, G, C//G, Nc), query for local/ non-local ops
           LocalKey : (B, G, C//G, Nc, K), key for local ops
           NonLocalKey : (B, G, C//G, Np, K), key for nonlocal ops
           idx_knn : (B, Nc, K), idx of knn for calculating attention centrality of current layer
           idx_nonlocal : (B, G, Np, K), idx of top-k important nodes of previous layer

        Returns:
           output features of query points: [B, C_out, 3]
        """ 

        # 1. Acquire Local Neighborhood index and relative position
        #idx, idx_mask = masked_ordered_ball_query(self.radius, self.nsample, query_xyz, support_xyz,
                                                  #query_mask, support_mask)

        # GT Module Properties
        # 1. query = support 
        QueryandKey, Value, AC = queryandkey, value, attention_centrality # B, C, Nc / B, C, Nc / B, G, Nc
        B = QueryandKey.shape[0]
        C = QueryandKey.shape[1]
        npoints = QueryandKey.shape[2]
        nsample = self.nsample
        groups = 9
        Cnl = 0
        Cl = C - Cnl

      
        # 0. Acquire Attention Centrality Index
        idx_knn = idx_knn.unsqueeze(1).repeat(1,groups,1,1) # B, G, Nc, K

        # 1. Local and Non-local Query 
        LocalQuery = QueryandKey[:,:Cl,:].contiguous().view(B, groups, Cl//groups, npoints) # B, G, C//G, Nc

        # 2. Key
        LocalKey = QueryandKey[:,:Cl,:].contiguous().view(B, groups, Cl//groups, npoints) # B, G, C//G, Nc
                
        LocalKey = self.batch_gather_neighbour(LocalKey.permute(0,1,3,2), idx_knn) # B, G, Nc, C//G and B, G, Nc, K -> B, G, C//G, Nc, K


        # 3. Relative Pos for Value
        #xyz_trans = support_xyz.transpose(1, 2).unsqueeze(1).repeat(1,groups,1,1).contiguous() # B,G,Nc,3
        #Lxyz = self.batch_gather_neighbour(xyz_trans.permute(0,1,3,2), idx_knn) # B,G,Nc,3 and B,G,Nc,K -> B,G,3,Nc,K
        #Lxyz -= query_xyz.transpose(1, 2).unsqueeze(1).unsqueeze(-1) # B,1,Nc,C,1

        # 4. Value
        LocalValue = Value[:,:Cl,:].contiguous().view(B, groups, Cl//groups, npoints)
        LocalValue = self.batch_gather_neighbour(LocalValue.permute(0,1,3,2), idx_knn) # B, G, Nc, C//G and B, G, Nc, K -> B, G, C//G, Nc, K

        if self.use_xyz:
            LocalValue = torch.cat([LocalValue, Lxyz], dim=2) 

        # 5. Attention Centrality
        LocalAtt = (LocalQuery.unsqueeze(-1) * LocalKey).sum(2) # B, G, N, K
        LocalAtt = F.softmax(LocalAtt, dim=-1) # B, G, N, K

        current_AC = scatter_sparse_attention_centrality(LocalAtt, idx_knn[:,0,:,:]) # B, G, N
        
        
        # 6. Positional Encoding
        '''
        if self.position_embedding == 'xyz':
            position_embedding = torch.unsqueeze(relative_position, 1)
            aggregation_features = neighborhood_features.view(B, C // 3, 3, npoint, self.nsample)
            aggregation_features = position_embedding * aggregation_features  # (B, C//3, 3, npoint, nsample)
            aggregation_features = aggregation_features.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
        elif self.position_embedding == 'sin_cos':
            feat_dim = C // 6
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(query_xyz.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * relative_position, -1)  # (B, 3, npoint, nsample, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, nsample, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, nsample, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
            position_embedding = position_embedding.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
            aggregation_features = neighborhood_features * position_embedding  # (B, C, npoint, nsample)
        else:
            pass
        '''

        # 7. Sum
        LocalFeature = (LocalAtt.unsqueeze(2) * LocalValue).sum(-1) # B, G, Cl//G+3, N 
        LocalFeature = LocalFeature.view(B, -1, npoints) # B, Cl+3G, N

        #if self.output_conv:
        #    out_features = self.out_conv(out_features)
        #else:

        #Features = self.out_transform(LocalFeatures)

        return LocalFeature, current_AC
    
    
    def gather_neighbour(self, pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1) # b, nk

        #TODO THIS LINE IS REQUIRED TO BE OPTIMIZED
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2])) # b,n,c -> b,nk,c
        
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features.permute(0,3,1,2).unsqueeze(1) # B,1,C//G,N,K


    def gather_neighbourv2(self, pc, neighbor_idx):
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]

        idx_base = torch.arange(0, batch_size, device='cuda').view(-1, 1, 1)*num_points
        neighbor_idx = neighbor_idx + idx_base
        neighbor_idx = neighbor_idx.view(-1)

        pc = pc.contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = pc.view(batch_size*num_points, -1)[neighbor_idx, :]
        feature = feature.view(batch_size, num_points, -1, d) 
        return feature.permute(0,3,1,2).unsqueeze(1)


    def batch_gather_neighbour(self, pc, neighbor_idx):  
        # pc: B, G, N, C
        # neighbor_idx: B, G, N, K

        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        groups = pc.shape[1]
        num_points = pc.shape[2]
        d = pc.shape[3]

        features = []
        for g in range(groups):
            g_pc = pc[:,g,:,:] # B, N, C
            g_neighbor_idx = neighbor_idx[:,g,:,:] # B, N, K
            feature = self.gather_neighbourv2(g_pc, g_neighbor_idx) # B, 1, C//G, N, K
            features.append(feature)

        features = torch.cat(features, dim=1)

        return features

        # matrix implementation
        '''
        pc = pc.reshape(batch_size*groups, num_points, d) # bg, n, c
        index_input = neighbor_idx.reshape(batch_size*groups, -1) # bg, nk
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2])) # bg,n,c -> bg,nk,c
        features = features.reshape(batch_size, groups, num_points, neighbor_idx.shape[-1], d)  # batch*groups*npoint*nsamples*channel
        return features.permute(0,1,4,2,3) # b,g,c//g,n,k
        '''



class GTModuleV1(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A GT operator for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(GTModuleV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.radius = radius. No Need for GTModule
        self.nsample = nsample
        self.position_embedding = config.position_embedding
        self.reduction = config.reduction
        self.output_conv = config.output_conv or (self.in_channels != self.out_channels)

        #self.use_xyz = True
        self.use_xyz = False

        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))


    def forward(self, query_xyz, support_xyz, query_mask, support_mask, queryandkey, value, attention_centrality=None, idx_knn=None):
        """
        Args:
           Query : (B, G, C//G, Nc), query for local/ non-local ops
           LocalKey : (B, G, C//G, Nc, K), key for local ops
           NonLocalKey : (B, G, C//G, Np, K), key for nonlocal ops
           idx_knn : (B, Nc, K), idx of knn for calculating attention centrality of current layer
           idx_nonlocal : (B, G, Np, K), idx of top-k important nodes of previous layer

        Returns:
           output features of query points: [B, C_out, 3]
        """ 

        # 1. Acquire Local Neighborhood index and relative position
        #idx, idx_mask = masked_ordered_ball_query(self.radius, self.nsample, query_xyz, support_xyz,
                                                  #query_mask, support_mask)


        # GT Module Properties
        # 1. query = support 
        QueryandKey, Value, AC = queryandkey, value, attention_centrality # B, C, Nc / B, C, Nc / B, G, Nc
        B = QueryandKey.shape[0]
        C = QueryandKey.shape[1]
        npoints = QueryandKey.shape[2]
        nsample = self.nsample
        groups = 9

        Cnl = C//4
        Cl = C - Cnl

        # 0. Acquire Attention Centrality Index
        _, idx_ac = AC.topk(k=nsample, dim=2) # B, G, Nc -> B, G, K
        idx_ac = idx_ac.unsqueeze(2).repeat(1,1,npoints,1) # B, G, Nc, K
        idx_knn = idx_knn.unsqueeze(1).repeat(1,groups,1,1) # B, G, Nc, K


        # 1. Local and Non-local Query 
        #print(QueryandKey[:,:Cl,:].shape)
        LocalQuery = QueryandKey[:,:Cl,:].contiguous().view(B, groups, Cl//groups, npoints) # B, G, C//G, Nc
        NonLocalQuery = QueryandKey[:,Cl:,:].contiguous().view(B, groups, Cnl//groups, npoints)

        # 2. Key
        LocalKey = QueryandKey[:,:Cl,:].contiguous().view(B, groups, Cl//groups, npoints) # B, G, C//G, Nc
        NonLocalKey = QueryandKey[:,Cl:,:].contiguous().view(B, groups, Cnl//groups, npoints) # B, G, C//G, Nc


        LocalKey = self.batch_gather_neighbour(LocalKey.permute(0,1,3,2), idx_knn) # B, G, Nc, C//G and B, G, Nc, K -> B, G, C//G, Nc, K
        NonLocalKey = self.batch_gather_neighbour(NonLocalKey.permute(0,1,3,2), idx_ac)

        # 3. Relative Pos for Value
        xyz_trans = support_xyz.transpose(1, 2).unsqueeze(1).repeat(1,groups,1,1).contiguous() # B,G,Nc,3
        #Lxyz = self.batch_gather_neighbour(xyz_trans.permute(0,1,3,2), idx_knn) # B,G,Nc,3 and B,G,Nc,K -> B,G,3,Nc,K
        #Lxyz -= query_xyz.transpose(1, 2).unsqueeze(1).unsqueeze(-1) # B,1,Nc,C,1
        #NLxyz = self.batch_gather_neighbour(xyz_trans.permute(0,1,3,2), idx_ac) # B,G,Nc,3 and B,G,Nc,K -> B,G,3,Nc,K
        #NLxyz -= query_xyz.transpose(1, 2).unsqueeze(1).unsqueeze(-1) # B,1,Nc,C,1

        # 4. Value
        LocalValue = Value[:,:Cl,:].contiguous().view(B, groups, Cl//groups, npoints)
        NonLocalValue = Value[:,Cl:,:].contiguous().view(B, groups, Cnl//groups, npoints)
        LocalValue = self.batch_gather_neighbour(LocalValue.permute(0,1,3,2), idx_knn) # B, G, Nc, C//G and B, G, Nc, K -> B, G, C//G, Nc, K
        NonLocalValue = self.batch_gather_neighbour(NonLocalValue.permute(0,1,3,2), idx_ac)

        if self.use_xyz:
            LocalValue = torch.cat([LocalValue, Lxyz], dim=2) 
            NonLocalValue = torch.cat([NonLocalValue, NLxyz], dim=2)

        # 5. Attention Centrality
        LocalAtt = (LocalQuery.unsqueeze(-1) * LocalKey).sum(2) # B, G, N, K
        LocalAtt = F.softmax(LocalAtt, dim=-1) # B, G, N, K

        NonLocalAtt = (NonLocalQuery.unsqueeze(-1) * NonLocalKey).sum(2) # B, G, N, K
        NonLocalAtt = F.softmax(NonLocalAtt, dim=-1) # B, G, N, K

        current_AC = scatter_sparse_attention_centrality(LocalAtt, idx_knn[:,0,:,:]) # B, G, N

        
        # 6. Positional Encoding
        '''
        if self.position_embedding == 'xyz':
            position_embedding = torch.unsqueeze(relative_position, 1)
            aggregation_features = neighborhood_features.view(B, C // 3, 3, npoint, self.nsample)
            aggregation_features = position_embedding * aggregation_features  # (B, C//3, 3, npoint, nsample)
            aggregation_features = aggregation_features.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
        elif self.position_embedding == 'sin_cos':
            feat_dim = C // 6
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(query_xyz.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * relative_position, -1)  # (B, 3, npoint, nsample, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, nsample, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, nsample, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
            position_embedding = position_embedding.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
            aggregation_features = neighborhood_features * position_embedding  # (B, C, npoint, nsample)
        else:
            pass
        '''

        # 7. Sum
        LocalFeature = (LocalAtt.unsqueeze(2) * LocalValue).sum(-1) # B, G, Cl//G+3, N 
        NonLocalFeature = (NonLocalAtt.unsqueeze(2) * NonLocalValue).sum(-1) # B, G, Cnl//G+3, N

        LocalFeature = LocalFeature.view(B, -1, npoints)
        NonLocalFeature = NonLocalFeature.view(B, -1, npoints)
        
        Features = torch.cat([LocalFeature, NonLocalFeature], dim=1)

        #if self.output_conv:
        #    out_features = self.out_conv(out_features)
        #else:

        #Features = self.out_transform(Features)

        return Features, current_AC


    def gather_neighbour(self, pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1) # b, nk
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2])) # b,n,c -> b,nk,c
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features.permute(0,3,1,2).unsqueeze(1)

    def gather_neighbourv2(self, pc, neighbor_idx):
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]

        idx_base = torch.arange(0, batch_size, device='cuda').view(-1, 1, 1)*num_points
        neighbor_idx = neighbor_idx + idx_base
        neighbor_idx = neighbor_idx.view(-1)

        pc = pc.contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = pc.view(batch_size*num_points, -1)[neighbor_idx, :]
        feature = feature.view(batch_size, num_points, -1, d) 
        return feature.permute(0,3,1,2).unsqueeze(1)

    def batch_gather_neighbour(self, pc, neighbor_idx):  
        # pc: B, G, N, C
        # neighbor_idx: B, G, N, K

        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        groups = pc.shape[1]
        num_points = pc.shape[2]
        d = pc.shape[3]

        features = []
        for g in range(groups):
            g_pc = pc[:,g,:,:] # B, N, C
            g_neighbor_idx = neighbor_idx[:,g,:,:] # B, N, K
            feature = self.gather_neighbourv2(g_pc, g_neighbor_idx) # B, 1, C//G, N, K
            features.append(feature)
        features = torch.cat(features, dim=1)

        return features

        # matrix implementation
        '''
        pc = pc.reshape(batch_size*groups, num_points, d) # bg, n, c
        index_input = neighbor_idx.reshape(batch_size*groups, -1) # bg, nk
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2])) # bg,n,c -> bg,nk,c
        features = features.reshape(batch_size, groups, num_points, neighbor_idx.shape[-1], d)  # batch*groups*npoint*nsamples*channel
        return features.permute(0,1,4,2,3) # b,g,c//g,n,k
        '''




class GTMaskedQueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, config=None):
        super(MaskedQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, queryandkey, value, attention_centrality):

        dim_value = value.shape[1]//4
        localvalue = value[:,dim_value:,:]
        nonlocalvalue = value[:,:dim_value,:] # B, Cnl, N

        batch = query_xyz[0]
        groups = 9
        npoint = query_xyz[2]
        dim_localvalue = value.shape[1]//4
        dim_nonlocalvalue = value.shape[1] - dim_localvalue
        nsample = 16

        # 1. Acquire Local Neighborhood index and relative position
        idx, idx_mask = masked_ordered_ball_query(self.radius, self.nsample, query_xyz, support_xyz,
                                                  query_mask, support_mask)

        xyz_trans = support_xyz.transpose(1, 2).contiguous()

        grouped_localxyz = grouping_operation(xyz_trans, idx)  # b,3,np and b,nc,k -> b,3,nc,k

        grouped_localxyz -= query_xyz.transpose(1, 2).unsqueeze(-1)

        if self.normalize_xyz:
            grouped_localxyz /= self.radius

        grouped_localvalue = grouping_operation(torch.cat([localvalue, queryandkey], dim=1), idx) # b,(f+1),np and b,nc,k -> b,f+1,nc,k
        #attention_centrality = grouped_localvalue[:,-groups:,:,0] # B, G, Nc
        grouped_localvalue = grouped_localvalue[:,:dim_localvalue,:,:] # B, Cl, Nc, K
        queryandkey = grouped_localvalue[:,dim_localvalue:-groups,:,0] # B, C, Nc

        grouped_localxyz = grouped_localxyz.view(batch, groups, 3, npoint, nsample)
        grouped_localvalue = grouped_localvalue.view(batch, groups, dim_localvalue, npoint, nsample) 

        #TODO must check, the first one is itself. 
        if self.use_xyz:
            new_localfeatures = torch.cat([grouped_localxyz, grouped_localvalue], dim=2)  # (B, G, C + 3, npoint, nsample)
        else:
            new_localfeatures = grouped_localvalue


        # 2. Acquire Non-local Neighborhood index and relative position
        _, idx_ac = attention_centrality.topk(k=nsample, dim=2) # B, G, Np -> B, G, K
        idx_ac = idx_ac.unsqueeze(2).repeat(1,1,npoint,1) # B, G, Nc, K

        new_nonlocalfeatures = []
        nonlocalvalue = nonlocalvalue.view(batch, groups, dim_nonlocalvalue, npoint) # B, G, C//G, N
        for j in range(8):
            grouped_nonlocalxyz = grouping_operation(xyz_trans, idx_ac[:,j,:,:])
            grouped_nonlocalxyz -= query_xyz.transpose(1, 2).unsqueeze(-1)
            grouped_nonlocalvalue = grouping_operation(nonlocalvalue[:,j,:,:], idx_ac[:,j,:,:])
            new_nonlocalfeature = torch.cat([grouped_nonlocalxyz, grouped_nonlocalvalue], dim=1).unsqueeze(1) # B, 1, C+3, N, K
            new_nonlocalfeatures.append(new_nonlocalfeature)
        new_nonlocalfeatures = torch.cat(new_nonlocalfeatures, dim=1) # B, G, C//G+3, N, K
            

        if self.ret_grouped_xyz:
            return new_localfeatures, grouped_localxyz, new_nonlocalfeatures, grouped_nonlocalxyz, idx_mask, queryandkey
        else:
            return new_localfeatures, new_nonlocalfeatures, idx_mask, queryandkey





class MaskedQueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, config=None):
        super(MaskedQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, features=None):
        idx, idx_mask = masked_ordered_ball_query(self.radius, self.nsample, query_xyz, support_xyz,
                                                  query_mask, support_mask)

        xyz_trans = support_xyz.transpose(1, 2).contiguous()

        grouped_xyz = grouping_operation(xyz_trans, idx)  # b,3,np and b,nc,k -> b,3,nc,k

        #grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            try:
                if config.local_aggregation_type in ['gtv1', 'gtv2']:
                    grouped_features = grouping_operation(features, idx) # b,(f+1),np and b,nc,k -> b,f+1,nc,k
                    #TODO must check, the first one is itself.
                    dim_features = grouped_features.shape[1]
                    dim_queryandkey = (dim_features-8)//2
                    attention_centrality = grouped_features[:,-8:,:,0]
                    queryandkey = grouped_features[:,dim_query:2*dim_query,:,0]
                    grouped_features = grouped_features[:,:dim_query,:,:]

            except:
                grouped_features = grouping_operation(features, idx) # b,(f+1),np and b,nc,k -> b,f+1,nc,k

            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (self.use_xyz), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        try:
            if config.local_aggregation_type in ['gtv1', 'gtv2']:
                if self.ret_grouped_xyz:
                    return new_features, grouped_xyz, idx_mask, attention_centrality, queryandkey
                else:
                    return new_features, idx_mask, attention_centrality, queryandkey
        except:
            if self.ret_grouped_xyz:
                return new_features, grouped_xyz, idx_mask
            else:
                return new_features, idx_mask


class MaskedMaxPool(nn.Module):
    def __init__(self, npoint, radius, nsample, sampleDl):
        super(MaskedMaxPool, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.sampleDl = sampleDl
        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True)

    def forward(self, xyz, mask, features):
        # sub sample
        sub_xyz, sub_mask = masked_grid_subsampling(xyz, mask, self.npoint, self.sampleDl)
        sub_xyz = sub_xyz.contiguous()
        sub_mask = sub_mask.contiguous()
        # masked ordered ball query
        neighborhood_features, grouped_xyz, idx_mask = self.grouper(sub_xyz, xyz, sub_mask, mask,
                                                                    features)  # (B, C, npoint, nsample)

        sub_features = F.max_pool2d(
            neighborhood_features, kernel_size=[1, neighborhood_features.shape[3]]
        )  # (B, C, npoint, 1)
        sub_features = torch.squeeze(sub_features, -1)  # (B, C, npoint)
        return sub_xyz, sub_mask, sub_features




# ATT AGG  ==============================================================================


class PosPool(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A PosPool operator for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(PosPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.nsample = nsample
        self.position_embedding = config.pospool.position_embedding
        self.reduction = config.pospool.reduction
        self.output_conv = config.pospool.output_conv or (self.in_channels != self.out_channels)

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=True)
        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """

        B = query_xyz.shape[0]
        C = support_features.shape[1]
        npoint = query_xyz.shape[1]
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask, support_mask, support_features)

        if self.position_embedding == 'xyz':
            position_embedding = torch.unsqueeze(relative_position, 1)
            aggregation_features = neighborhood_features.view(B, C // 3, 3, npoint, self.nsample)
            aggregation_features = position_embedding * aggregation_features  # (B, C//3, 3, npoint, nsample)
            aggregation_features = aggregation_features.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
        elif self.position_embedding == 'sin_cos':
            feat_dim = C // 6
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(query_xyz.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * relative_position, -1)  # (B, 3, npoint, nsample, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, nsample, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, nsample, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
            position_embedding = position_embedding.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
            aggregation_features = neighborhood_features * position_embedding  # (B, C, npoint, nsample)
        else:
            raise NotImplementedError(f'Position Embedding {self.position_embedding} not implemented in PosPool')

        if self.reduction == 'max':
            out_features = F.max_pool2d(
                aggregation_features, kernel_size=[1, self.nsample]
            )
            out_features = torch.squeeze(out_features, -1)
        elif self.reduction == 'avg' or self.reduction == 'mean':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
            neighborhood_num = feature_mask.sum(-1)
            out_features /= neighborhood_num
        elif self.reduction == 'sum':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
        else:
            raise NotImplementedError(f'Reduction {self.reduction} not implemented in PosPool ')

        if self.output_conv:
            out_features = self.out_conv(out_features)
        else:
            out_features = self.out_transform(out_features)

        return out_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features



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


