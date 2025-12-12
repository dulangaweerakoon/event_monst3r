# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from dust3r.heads.postprocess import postprocess
import dust3r.utils.path_to_croco  # noqa: F401
from models.dpt_block import DPTOutputAdapter, Interpolate  # noqa


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # print(path_1.shape)

        # Output head
        out = self.head(path_1)

        # print("DPT head output shape:", out.shape)

        return out


class ObjectAwareDPTOutputAdapter(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768, last_dim=32):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

        self.last_dim = last_dim

        # self.interpolate = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        self.interpolate = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        # design a mean head to predict the mean depth for each object with fully connected layers
        self.mean_head = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(True),
            nn.Linear(self.feature_dim // 4, 3)
        )

        # print(self.num_channels)
        # while(True):
        #     continue

        self.head = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=3, stride=1, padding=1),
            # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(self.feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0)
        )
    def attention_pooling(self, x, mask):
        """
        x: B x C x H x W
        mask: B x 1 x H x W
        """
        B, C, H, W = x.shape

        # expand x to have multiple masks
        x = x.unsqueeze(1)  # B x 1 x C x H x W
        mask = mask.unsqueeze(2)  # B x num_masks x 1 x H x W
        x = x.expand(-1, mask.shape[1], -1, -1, -1)  # B x num_masks x C x H x W
        mask = mask.expand(-1, -1, C, -1, -1)

        x = x * mask  # B x num_mask x C x H x W
        sum_x = x.sum(dim=(3, 4))  # B x num_mask x C
        sum_mask = mask.sum(dim=(3, 4)) + 1e-6  # B x num_mask x C
        pooled = sum_x / sum_mask  # B x num_mask x C
        return pooled
    
    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None, attention_masks=None, max_instance_ids=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        path_1 = self.interpolate(path_1)

        # attention pooling here
        pooled_feat = self.attention_pooling(path_1, attention_masks) # B x num_masks x C

        # predict mean depth for each object
        # print("pooled feat shape", pooled_feat.shape, path_1.shape)
        mean_depths = self.mean_head(pooled_feat)  # B x num_masks x 3

        # create a tensor of shape B x num_masks x 3 x H x W to add to path_1
        B, num_masks, _ = mean_depths.shape

        # create a zero tensor of shape B x num_masks x 3 x H x W
        mean_depths_expanded = mean_depths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, path_1.shape[2], path_1.shape[3])  # B x num_masks x 3 x H x W
        # print("mean depths expanded shape:", mean_depths_expanded.shape)

        # create a mean_depth_map of shape B x num_masks x 3 x H x W by multiplying mean_depths_expanded with attention_masks
        attention_masks_expanded = attention_masks.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # B x num_masks x 3 x H x W
        mean_depth_map = mean_depths_expanded * attention_masks_expanded  # B x num_masks x 3 x H x W


        # expand pooled features to B x num_masks x C x H x W
        pooled_feat_expanded = pooled_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, path_1.shape[1], path_1.shape[2], path_1.shape[3])  # B x num_masks x C x H x W
        # concatenate path_1 with pooled_feat_expanded along channel dimension
        path_1_expanded = path_1.unsqueeze(1).expand(-1, num_masks, -1, -1, -1)  # B x num_masks x C x H x W
        # concat along channel dimension
        path_1 = torch.cat([path_1_expanded, pooled_feat_expanded], dim=2)  # B x num_masks x (C + C) x H x W
        # Output head
        path_1 = path_1.reshape(B * num_masks, path_1.shape[2], path_1.shape[3], path_1.shape[4])  # (B * num_masks) x (C + C) x H x W
        out_masked = self.head(path_1)
        out_masked = out_masked.reshape(B, num_masks, out_masked.shape[1], out_masked.shape[2], out_masked.shape[3])  # B x num_masks x num_channels x H x W

        # add mean depth map to out
        # print("out masked shape before adding mean depth:", out_masked.shape)
        out_masked[:,:,0:3,:,:] = out_masked[:,:,0:3,:,:] + mean_depth_map

        # get the output for each pixel by adding over all masks
        out = out_masked.sum(dim=1)  # B x num_channels x H x W

        # print(out.shape)

        # print(out.shape)

        # save auxilarry out_masked with B*num_masksxnum_channelsxHxW

        out_masked = out_masked.reshape(B * num_masks, out_masked.shape[2], out_masked.shape[3], out_masked.shape[4])  # (B * num_masks) x num_channels x H x W
        attention_masks_expanded = attention_masks_expanded.reshape(B * num_masks, attention_masks_expanded.shape[2], attention_masks_expanded.shape[3], attention_masks_expanded.shape[4])  # (B * num_masks) x 1 x H x W
        attention_masks_expanded = attention_masks_expanded[:,0,:,:]  # keep only one channel
        
        # print("Before: ",out.shape, out_masked.shape, attention_masks_expanded.shape)
        return out, out_masked, attention_masks_expanded

class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        # print("DPT head raw output shape:", out.shape)
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        # print("DPT head output shape:", out.keys())
        return out
    

class ObjectAwareWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(ObjectAwareWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = ObjectAwareDPTOutputAdapter(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info, attention_masks=None, max_instance_ids=None):
        # print("Max IDs: ",max_instance_ids)
        out, out_masked, attention_expanded = self.dpt(x, image_size=(img_info[0], img_info[1]), attention_masks=attention_masks, max_instance_ids=max_instance_ids)
        # print("DPT head raw output shape:", out.shape)
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
            out_masked = self.postprocess(out_masked, self.depth_mode, self.conf_mode)
        # print("Object Aware DPT head output shape:", out.keys())
        return out, out_masked, attention_expanded



def create_dpt_head(net, has_conf=False,object_aware=False):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    if object_aware:
        return ObjectAwareWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='regression')
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='regression')
