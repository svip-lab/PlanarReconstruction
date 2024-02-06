import torch
import torch.nn as nn

from models import resnet_scene as resnet
from transformers import DPTModel, DPTConfig, DPTImageProcessor, DPTForSemanticSegmentation
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union


class ResNet(nn.Module):
    def __init__(self, orig_resnet):
        super(ResNet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1

        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2

        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x1 = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5 
    

    
# def _get_backbone_hidden_size(config):
#     if config.backbone_config is not None and config.is_hybrid is False:
#         return config.backbone_config.hidden_size
#     else:
#         return config.hidden_size

    
# class DPTReassembleLayer(nn.Module):
#     def __init__(self, config, channels, factor):
#         super().__init__()
#         # projection
#         hidden_size = _get_backbone_hidden_size(config)
#         self.projection = nn.Conv2d(in_channels=hidden_size, out_channels=channels, kernel_size=1)

#         # up/down sampling depending on factor
#         if factor > 1:
#             self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
#         elif factor == 1:
#             self.resize = nn.Identity()
#         elif factor < 1:
#             # so should downsample
#             self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=int(1 / factor), padding=1)

#     def forward(self, hidden_state):
#         hidden_state = self.projection(hidden_state)
#         hidden_state = self.resize(hidden_state)
#         return hidden_state
    
# class DPTReassembleStage(nn.Module):
#     """
#     This class reassembles the hidden states of the backbone into image-like feature representations at various
#     resolutions.

#     This happens in 3 stages:
#     1. Map the N + 1 tokens to a set of N tokens, by taking into account the readout ([CLS]) token according to
#        `config.readout_type`.
#     2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
#     3. Resizing the spatial dimensions (height, width).

#     Args:
#         config (`[DPTConfig]`):
#             Model configuration class defining the model architecture.
#     """

#     def __init__(self, config):
#         super().__init__()

#         self.config = config
#         self.layers = nn.ModuleList()
#         if config.is_hybrid:
#             self._init_reassemble_dpt_hybrid(config)
#         else:
#             self._init_reassemble_dpt(config)

#         self.neck_ignore_stages = config.neck_ignore_stages

#     def _init_reassemble_dpt_hybrid(self, config):
#         r""" "
#         For DPT-Hybrid the first 2 reassemble layers are set to `nn.Identity()`, please check the official
#         implementation: https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/vit.py#L438
#         for more details.
#         """
#         for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
#             if i <= 1:
#                 self.layers.append(nn.Identity())
#             elif i > 1:
#                 self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

#         if config.readout_type != "project":
#             raise ValueError(f"Readout type {config.readout_type} is not supported for DPT-Hybrid.")

#         # When using DPT-Hybrid the readout type is set to "project". The sanity check is done on the config file
#         self.readout_projects = nn.ModuleList()
#         hidden_size = _get_backbone_hidden_size(config)
#         for i in range(len(config.neck_hidden_sizes)):
#             if i <= 1:
#                 self.readout_projects.append(nn.Sequential(nn.Identity()))
#             elif i > 1:
#                 self.readout_projects.append(
#                     nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
#                 )

#     def _init_reassemble_dpt(self, config):
#         for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
#             self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

#         if config.readout_type == "project":
#             self.readout_projects = nn.ModuleList()
#             hidden_size = _get_backbone_hidden_size(config)
#             for _ in range(len(config.neck_hidden_sizes)):
#                 self.readout_projects.append(
#                     nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
#                 )

#     def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
#         """
#         Args:
#             hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
#                 List of hidden states from the backbone.
#         """
#         out = []

#         for i, hidden_state in enumerate(hidden_states):
#             if i not in self.neck_ignore_stages:
#                 # reshape to (batch_size, num_channels, height, width)
#                 cls_token, hidden_state = hidden_state[:, 0], hidden_state[:, 1:]
#                 batch_size, sequence_length, num_channels = hidden_state.shape
#                 if patch_height is not None and patch_width is not None:
#                     hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, num_channels)
#                 else:
#                     size = int(math.sqrt(sequence_length))
#                     hidden_state = hidden_state.reshape(batch_size, size, size, num_channels)
#                 hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

#                 feature_shape = hidden_state.shape
#                 if self.config.readout_type == "project":
#                     # reshape to (batch_size, height*width, num_channels)
#                     hidden_state = hidden_state.flatten(2).permute((0, 2, 1))
#                     readout = cls_token.unsqueeze(1).expand_as(hidden_state)
#                     # concatenate the readout token to the hidden states and project
#                     hidden_state = self.readout_projects[i](torch.cat((hidden_state, readout), -1))
#                     # reshape back to (batch_size, num_channels, height, width)
#                     hidden_state = hidden_state.permute(0, 2, 1).reshape(feature_shape)
#                 elif self.config.readout_type == "add":
#                     hidden_state = hidden_state.flatten(2) + cls_token.unsqueeze(-1)
#                     hidden_state = hidden_state.reshape(feature_shape)
#                 hidden_state = self.layers[i](hidden_state)
#             out.append(hidden_state)

#         return out

    

class Baseline(nn.Module):
    def __init__(self, cfg):
        super(Baseline, self).__init__()
        self.arch = cfg.arch
        self.semantic = cfg.semantic
        if cfg.arch == 'dpt':
            self.arch = 'dpt'
            self.dpt_config = DPTConfig(image_size=256)
            self.dpt =  DPTForSemanticSegmentation(config = self.dpt_config).from_pretrained("Intel/dpt-large-ade")
            self.dpt_config = self.dpt.config
            self.dpt_proj = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.dpt_conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
            self.dpt_up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.dpt_conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            self.dpt_maxp = nn.MaxPool2d(kernel_size=(9,1), dilation=(8,1), stride = (1,1)) 
            self.dpt_relu = nn.ReLU()

            self.dpt_head = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=(9,1), dilation=(8,1), stride = (1,1)) ,
                nn.ReLU()
            )
        else:
            orig_resnet = resnet.__dict__[cfg.arch](pretrained=cfg.pretrained)
            self.backbone = ResNet(orig_resnet)
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.relu = nn.ReLU(inplace=True)

        channel = 64
        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv2 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv1 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv0 = nn.Conv2d(channel, channel, (1, 1))

        # lateral
        self.c5_conv = nn.Conv2d(2048, channel, (1, 1))
        self.c4_conv = nn.Conv2d(1024, channel, (1, 1))
        self.c3_conv = nn.Conv2d(512, channel, (1, 1))
        self.c2_conv = nn.Conv2d(256, channel, (1, 1))
        self.c1_conv = nn.Conv2d(128, channel, (1, 1))

        self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)


        

        # plane or non-plane classifier
        self.pred_prob = nn.Conv2d(channel, 1, (1, 1), padding=0)
        # embedding
        self.embedding_conv = nn.Conv2d(channel, 2, (1, 1), padding=0)
        # depth prediction
        self.pred_depth = nn.Conv2d(channel, 1, (1, 1), padding=0)
        # surface normal prediction
        self.pred_surface_normal = nn.Conv2d(channel, 3, (1, 1), padding=0)
        # surface plane parameters
        self.pred_param = nn.Conv2d(channel, 3, (1, 1), padding=0)
        
        if cfg.semantic:
            # semantic segmentation
            self.pred_semantic = nn.Conv2d(channel, 41, (1, 1), padding=0)
            # combination for semantic pool
            self.combination = nn.Conv2d(43, 2, (1, 1), padding=0)

    def top_down(self, x):
        c1, c2, c3, c4, c5 = x

        p5 = self.relu(self.c5_conv(c5))
        p4 = self.up_conv5(self.upsample(p5)) + self.relu(self.c4_conv(c4))
        p3 = self.up_conv4(self.upsample(p4)) + self.relu(self.c3_conv(c3))
        p2 = self.up_conv3(self.upsample(p3)) + self.relu(self.c2_conv(c2))
        p1 = self.up_conv2(self.upsample(p2)) + self.relu(self.c1_conv(c1))

        p0 = self.upsample(p1)

        p0 = self.relu(self.p0_conv(p0))

        return p0, p1, p2, p3, p4, p5
    
    def dpt_backbone(self, x):
        
        l1 = self.dpt.dpt(x, output_hidden_states=True, output_attentions=True, return_dict=True)
        hidden_states = l1.hidden_states
        # print(l1.shape)
        # print('dpt_enc'*10)
        # output_attentions = l1.output_attentions
        # return_dict = l1.return_dict
        hidden_states = [ feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.dpt.config.backbone_out_indices]
        
        l2 = self.dpt.neck(hidden_states=hidden_states)
        l3 = l2[-1]
        # print(l3.shape)
        # print('dpt_dec_'*20)
        l4 = self.dpt_head(l3)
        # print(l4.shape)
        # print('dpt_head_'*20)
        return l1,l2,l3,l4

    def forward(self, x):

        # Garbage, can delete
        # print(x.size())
        # print('11'*111)
        # print(x[0])
        # feature_extractor = dpt.from_pretrained("Intel/dpt-large")
        # test = self.feature_extractor(x,do_resize=False,return_tensors='pt')
        # print(test.data.keys())
        # print('x'*100)
        # print(test.data['pixel_values'][0])
        # print('00'*111)
        # print(test.data['pixel_values'].size())
        # print('00'*111)
        
        if self.arch == 'dpt':
            p3,p2,p1,p0 = self.dpt_backbone(x)
        else:
            # bottom up
            c1, c2, c3, c4, c5 = self.backbone(x)
            # print('_'*100)
            # print(c1.size(),'_',c2.size(),'_',c3.size(),'_',c4.size(),'_',c5.size())
            # print('..'*100)

            # top down
            p0, p1, p2, p3, p4, p5 = self.top_down((c1, c2, c3, c4, c5))
            # print(p0.size(),'_',p1.size(),'_',p2.size(),'_',p3.size(),'_',p4.size(),'_',p5.size())
            # print('='*100)
            

        # output
        
        prob = self.pred_prob(p0)
        embedding = self.embedding_conv(p0)
        depth = self.pred_depth(p0)
        surface_normal = self.pred_surface_normal(p0)
        param = self.pred_param(p0)
        
        if self.semantic:
            semantic = self.pred_semantic(p0)
            combination = self.combination(torch.cat((embedding, semantic), dim=1))
            return prob, embedding, depth, surface_normal, param, semantic, combination
            
        return prob, embedding, depth, surface_normal, param

    
