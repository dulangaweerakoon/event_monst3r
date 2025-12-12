# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

import cv2

import matplotlib.pyplot as plt
import random
import numpy as np

from PIL import Image

import numpy as np

from transformers import pipeline

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape, transpose_to_landscape_ob_aware
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed, ManyAR_PatchEmbed
from third_party.raft import load_RAFT
from dust3r.event_rgb import EventRGBFusionBlock, CrossAttentionFusion
from dust3r.aggregrator import Aggregator, EventAggregator

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2



class AsymmetricCroCo3DEventStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        # Event encoder convolutional to 1,H,W -> 3,H,W
        self.event_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),   
        )
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['event']#.unsqueeze(1)  # B,1,H,W
        img2 = view2['event']#.unsqueeze(1)  # B,1,H,W
        B = img1.shape[0]

        img1 = self.event_conv(img1)  # B,3,H,W
        img2 = self.event_conv(img2)  # B,3,H,W
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)
        
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)

        # while (True):
        #     continue

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2



class AsymmetricCroCo3DEventRGBStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        # Event RGB encoder convolutional to 4,H,W -> 3,H,W
        self.event_conv = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),   
        )
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        ev1 = view1['event']#.unsqueeze(1)  # B,1,H,W
        ev2 = view2['event']#.unsqueeze(1)  # B,1,H,W

        img1 = view1['img']  # B,3,H,W
        img2 = view2['img']  # B,3,H,W

        # stack event and image along channel dimension
        img1 = torch.cat((ev1, img1), dim=1)  # B,4,H,W
        img2 = torch.cat((ev2, img2), dim=1)  # B,4,H,W

        B = img1.shape[0]

        img1 = self.event_conv(img1)  # B,3,H,W
        img2 = self.event_conv(img2)  # B,3,H,W
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)
        
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)

        # while (True):
        #     continue

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2


class AsymmetricCroCo3DEventRGBStereoV2 (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        # Event RGB encoder convolutional to 4,H,W -> 3,H,W
        self.event_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),   
        )

        self.ev_rgb_fusion = EventRGBFusionBlock(1024)  # num_patches, embed_dim
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)
        self.ev_patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, ev, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        ev_x, ev_pos = self.ev_patch_embed(ev, true_shape=true_shape)
        x = self.ev_rgb_fusion(x, ev_x)
        #concat x, ev_x along emb dimension

        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, ev1, ev2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((ev1, ev2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, ev1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, ev2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        ev1 = view1['event']#.unsqueeze(1)  # B,1,H,W
        ev2 = view2['event']#.unsqueeze(1)  # B,1,H,W

        img1 = view1['img']  # B,3,H,W
        img2 = view2['img']  # B,3,H,W

        # # stack event and image along channel dimension
        # img1 = torch.cat((ev1, img1), dim=1)  # B,4,H,W
        # img2 = torch.cat((ev2, img2), dim=1)  # B,4,H,W

        B = img1.shape[0]

        ev1 = self.event_conv(ev1)  # B,3,H,W
        ev2 = self.event_conv(ev2)  # B,3,H,W
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)
        
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)

        # while (True):
        #     continue

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], ev1[::2],ev2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, ev1[::2],ev2[::2], shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        print(res1['pts3d'].shape, res2.keys())
        return res1, res2
    



# VGGT Style EventRGB model


class VGGTEventRGBStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # print(croco_kwargs)

        self.aggregator = Aggregator(patch_size=croco_kwargs.get('patch_size',16))

        # self.ev_aggregator = Aggregator(patch_size=croco_kwargs.get('patch_size',16))

        # dust3r specific initialization
        # Event RGB encoder convolutional to 4,H,W -> 3,H,W
        self.event_conv = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),   
        )
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

        self.enc_blocks = None
        self.enc_norm = None


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2,B,emb_shape=1024,patch_start_idx=5):
        img1_ = img1.unsqueeze(1)  # B,1,3,H,W
        img2_ = img2.unsqueeze(1)  # B,1,3,H,W
        img1_2 = torch.cat((img1_, img2_), dim=1)
        # if img1.shape[-2:] == img2.shape[-2:]:
        #     out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
        #                                      torch.cat((true_shape1, true_shape2), dim=0))
        #     out, out2 = out.chunk(2, dim=0)
        #     pos, pos2 = pos.chunk(2, dim=0)
        # else:
        #     out, pos, _ = self._encode_image(img1, true_shape1)
        #     out2, pos2, _ = self._encode_image(img2, true_shape2)
        
        aggregated_tokens_list, _, pos_ = self.aggregator(img1_2)
        out_ = aggregated_tokens_list[-1]
        out = out_[:,0,patch_start_idx:]
        out2 = out_[:,0,patch_start_idx:]
        pos = pos_[:B,patch_start_idx:,:]
        pos2 = pos_[B:,patch_start_idx:,:]

        # print(out.shape, out2.shape, pos.shape, pos2.shape)

        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        ev1 = view1['event']#.unsqueeze(1)  # B,1,H,W
        ev2 = view2['event']#.unsqueeze(1)  # B,1,H,W

        img1 = view1['img']  # B,3,H,W
        img2 = view2['img']  # B,3,H,W

        # stack event and image along channel dimension
        img1 = torch.cat((ev1, img1), dim=1)  # B,4,H,W
        img2 = torch.cat((ev2, img2), dim=1)  # B,4,H,W

        B = img1.shape[0]

        img1 = self.event_conv(img1)  # B,3,H,W
        img2 = self.event_conv(img2)  # B,3,H,W
        # print(img1.shape, img2.shape)
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)
        
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)

        # aggregrate img1 and img2 in B, S, C, H, W in S dimension

        # aggregated_tokens_list, _, pos_ = self.aggregator(img1_2)

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        # if is_symmetrized(view1, view2):
        #     # computing half of forward pass!'
        #     feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
        #     feat1, feat2 = interleave(feat1, feat2)
        #     pos1, pos2 = interleave(pos1, pos2)
        # else:
        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2, B=B)
        
        # print("feat shape before agg", feat1.shape, feat2.shape, pos1.shape, pos2.shape)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
        
        # print("res", res1['pts3d'].shape, res2['pts3d'].shape)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2




class VGGTEventRGBStereoV2 (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # print(croco_kwargs)

        self.aggregator = Aggregator(patch_size=croco_kwargs.get('patch_size',16))
        # self.ev_aggregator = Aggregator(patch_size=croco_kwargs.get('patch_size',16))

        # self.ev_aggregator = Aggregator(patch_size=croco_kwargs.get('patch_size',16))

        # dust3r specific initialization
        # Event RGB encoder convolutional to 4,H,W -> 3,H,W
        self.event_conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),   
        )

        self.cross_attn = CrossAttentionFusion(embed_dim=2048, num_heads=croco_kwargs.get('num_heads',16))

        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

        self.enc_blocks = None
        self.enc_norm = None


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, ev1, ev2, true_shape1, true_shape2,B,emb_shape=1024,patch_start_idx=5):
        img1_ = img1.unsqueeze(1)  # B,1,3,H,W
        img2_ = img2.unsqueeze(1)  # B,1,3,H,W
        img1_2 = torch.cat((img1_, img2_), dim=1)

        ev1_ = ev1.unsqueeze(1)  # B,1,3,H,W
        ev2_ = ev2.unsqueeze(1)  # B,1,3,H,W
        ev1_2 = torch.cat((ev1_, ev2_), dim=1)
        # print("ev1_2 shape", ev1_2.shape)
        # if img1.shape[-2:] == img2.shape[-2:]:
        #     out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
        #                                      torch.cat((true_shape1, true_shape2), dim=0))
        #     out, out2 = out.chunk(2, dim=0)
        #     pos, pos2 = pos.chunk(2, dim=0)
        # else:
        #     out, pos, _ = self._encode_image(img1, true_shape1)
        #     out2, pos2, _ = self._encode_image(img2, true_shape2)
        
        aggregated_tokens_list, _, pos_ = self.aggregator(img1_2)
        out_ = aggregated_tokens_list[-1]
        out = out_[:,0,patch_start_idx:]
        out2 = out_[:,0,patch_start_idx:]
        pos = pos_[:B,patch_start_idx:,:]
        pos2 = pos_[B:,patch_start_idx:,:]

        ev_aggregated_tokens_list, _, ev_pos_ = self.aggregator(ev1_2)
        ev_out_ = ev_aggregated_tokens_list[-1]
        ev_out = ev_out_[:,0,patch_start_idx:]
        ev_out2 = ev_out_[:,0,patch_start_idx:]
        ev_pos = ev_pos_[:B,patch_start_idx:,:]
        ev_pos2 = ev_pos_[B:,patch_start_idx:,:]

        out = self.cross_attn(out, ev_out, pos_q=pos, pos_k=ev_pos)
        out2 = self.cross_attn(out2, ev_out2, pos_q=pos2, pos_k=ev_pos2)

        # print(out.shape, out2.shape, pos.shape, pos2.shape)

        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        ev1 = view1['event']#.unsqueeze(1)  # B,1,H,W
        ev2 = view2['event']#.unsqueeze(1)  # B,1,H,W

        img1 = view1['img']  # B,3,H,W
        img2 = view2['img']  # B,3,H,W

        # stack event and image along channel dimension
        # img1 = torch.cat((ev1, img1), dim=1)  # B,4,H,W
        # img2 = torch.cat((ev2, img2), dim=1)  # B,4,H,W

        B = img1.shape[0]

        ev1 = self.event_conv(ev1)  # B,3,H,W
        ev2 = self.event_conv(ev2)  # B,3,H,W
        # print(img1.shape, img2.shape)
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)
        
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)

        # aggregrate img1 and img2 in B, S, C, H, W in S dimension

        # aggregated_tokens_list, _, pos_ = self.aggregator(img1_2)

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        # if is_symmetrized(view1, view2):
        #     # computing half of forward pass!'
        #     feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
        #     feat1, feat2 = interleave(feat1, feat2)
        #     pos1, pos2 = interleave(pos1, pos2)
        # else:
        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, ev1, ev2, shape1, shape2, B=B)
        
        # print("feat shape before agg", feat1.shape, feat2.shape, pos1.shape, pos2.shape)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
        
        # print("res", res1['pts3d'].shape, res2['pts3d'].shape)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
    


# Event only VGGT style model 

class EventVGGTStereoV1 (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 num_bins=5,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        self.num_bins = num_bins

        # dust3r specific initialization
        # Event RGB encoder convolutional to 4,H,W -> 3,H,W
        self.event_conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_bins, 32, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),   
        )
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        ev1 = view1['event']#.unsqueeze(1)  # B,1,H,W
        ev2 = view2['event']#.unsqueeze(1)  # B,1,H,W

        # img1 = view1['img']  # B,3,H,W
        # img2 = view2['img']  # B,3,H,W

        # stack event and image along channel dimension
        # img1 = torch.cat((ev1, img1), dim=1)  # B,4,H,W
        # img2 = torch.cat((ev2, img2), dim=1)  # B,4,H,W

        B = ev1.shape[0]

        # print("ev1 shape", ev1.shape)
        # print("ev2 shape", ev2.shape)

        # print(view1['img'].shape) 

        img1 = self.event_conv(ev1)  # B,3,H,W
        img2 = self.event_conv(ev2)  # B,3,H,W
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)
        
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)

        # while (True):
        #     continue

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
    



class ObjectAwareDepthV1 (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 num_bins=5,
                 use_sam3=True,
                 sam_device=None,
                 sam_points_per_batch=128,
                 sam_pred_iou_thresh=0.7,
                #  sam_min_mask_region_area=200,
                sam_stability_score_thresh=0.7,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        self.num_bins = num_bins

        self.use_sam3 = use_sam3
        if self.use_sam3:
            if sam_device is None:
                sam_device = 0 if torch.cuda.is_available() else -1
            self.sam_device = sam_device
            self.sam_points_per_batch = sam_points_per_batch
            self.sam_pred_iou_thresh = sam_pred_iou_thresh
            self.sam_stability_score_thresh = sam_stability_score_thresh

            # This loads the SAM3 tracker based mask-generation pipeline
            self.sam3_generator = pipeline(
                "mask-generation",
                model="facebook/sam3",
                device=self.sam_device,
            )

        # dust3r specific initialization
        # Event RGB encoder convolutional to 4,H,W -> 3,H,W
        self.event_conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_bins, 32, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),   
        )
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), object_aware=True)
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), object_aware=True)
        # magic wrapper
        self.head1 = transpose_to_landscape_ob_aware(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape_ob_aware(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        ev1 = view1['event']#.unsqueeze(1)  # B,1,H,W
        ev2 = view2['event']#.unsqueeze(1)  # B,1,H,W

        # img1 = view1['img']  # B,3,H,W
        # img2 = view2['img']  # B,3,H,W

        # stack event and image along channel dimension
        # img1 = torch.cat((ev1, img1), dim=1)  # B,4,H,W
        # img2 = torch.cat((ev2, img2), dim=1)  # B,4,H,W

        B = ev1.shape[0]

        # print("ev1 shape", ev1.shape)
        # print("ev2 shape", ev2.shape)

        # print(view1['img'].shape) 

        img1 = self.event_conv(ev1)  # B,3,H,W
        img2 = self.event_conv(ev2)  # B,3,H,W
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)
        
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)

        # while (True):
        #     continue

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape, attention_masks=None, max_instance_ids=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        out, out_mask, attention_masks = head(decout, img_shape, attention_masks=attention_masks, max_instance_ids=max_instance_ids)
        return out, out_mask, attention_masks
    
    def _run_sam3_instance_ids(self, rgb_tensor):
        """
        rgb_tensor: [B, 3, H, W] torch float in [0,1] (assumed)
        returns: instance_ids [B, H, W] int32 (0 = background, 1..K = objects)
        """
        if not self.use_sam3:
            raise RuntimeError("SAM3 is not enabled. Set use_sam3=True when constructing the model.")

        B, C, H, W = rgb_tensor.shape
        device = rgb_tensor.device
        id_maps = []
        img_pils = []

        for b in range(B):
            img = rgb_tensor[b]  # [3, H, W]
            # to uint8 HxWx3
            img_np = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            img_np = (img_np * 255.0).astype("uint8")
            img_pil = Image.fromarray(img_np)
            img_pils.append(img_pil)


        with torch.no_grad():
            outputs = self.sam3_generator(
                img_pils,
                points_per_batch=self.sam_points_per_batch,
                pred_iou_thresh=self.sam_pred_iou_thresh,
                # min_mask_region_area=self.sam_min_mask_region_area,
                stability_score_thresh=self.sam_stability_score_thresh,
            )
        for b in range(B):
            masks_out = outputs[b]["masks"]
            # print("Masks shape", masks_out.shape)

            # masks_out can be a list of arrays or a single array
            if isinstance(masks_out, np.ndarray):
                if masks_out.ndim == 2:
                    masks_list = [masks_out]
                else:  # [N, H, W]
                    masks_list = [m for m in masks_out]
            else:
                masks_list = list(masks_out)

            id_map = np.zeros((H, W), dtype=np.int32)

            for idx, m in enumerate(masks_list, start=1):
                # some pipelines return dicts with "segmentation"
                if isinstance(m, dict) and "segmentation" in m:
                    m = m["segmentation"]
                m = np.array(m)
                if m.shape != (H, W):
                    # resize if needed
                    from PIL import Image as PILImage
                    m = np.array(
                        PILImage.fromarray(m).resize((W, H), resample=PILImage.NEAREST)
                    )
                mask_bool = m > 0
                id_map[mask_bool] = idx

            id_maps.append(torch.from_numpy(id_map))

        id_maps = torch.stack(id_maps, dim=0).to(device)  # [B, H, W]
        # print(f"SAM3 generated {id_maps[0].max().item()} instance IDs.")

        # create attention masks from id maps
        max_instance_id = id_maps.max().item()

        attention_masks = torch.zeros((B, max_instance_id + 1, H, W), dtype=torch.bool, device=device)
        per_batch_max_instance_ids = id_maps.view(B, -1).max(dim=1).values  # [B]
        for b in range(B):
            for inst_id in range(0, max_instance_id + 1):
                attention_masks[b, inst_id] = (id_maps[b] == inst_id)


        # temporarily save to visualize overlay on image
        # color_map = plt.get_cmap('tab20')
        # overlay = np.zeros((H, W, 3), dtype=np.float32)
        # id_map_np = id_maps[0].cpu().numpy()
        # for inst_id in range(1, id_map_np.max() + 1):
        #     color = color_map(random.randint(0, 19))[:3]  # RGB
        #     mask = id_map_np == inst_id
        #     overlay[mask] = color
        # img_np = rgb_tensor[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        # blended = (0.5 * img_np + 0.5 * overlay).clip(0, 1)
        # plt.imsave("sam3_instance_id_overlay.png", blended)
        # print("Saved SAM3 instance ID overlay to sam3_instance_id_overlay.png")
        return attention_masks, id_maps, per_batch_max_instance_ids

    def forward(self, view1, view2):
        # encode the two images --> B,S,D

        # print("Batch size in forward:", view1['img'].shape[0])

        # === NEW: if no instance_ids are provided, and SAM3 is enabled, run SAM3 on RGB ===
        # Expect view['img'] as [B, 3, H, W] if you want this path
        if self.use_sam3:
            # if 'instance_ids' not in view1 and 'img' in view1:
            view1_attention_masks, _, view1_max_ids = self._run_sam3_instance_ids(view1['img'])
            # if 'instance_ids' not in view2 and 'img' in view2:
            view2_attention_masks, _, view2_max_ids= self._run_sam3_instance_ids(view2['img'])
        # === END NEW ===
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1, res1_masked, res1_attention_masks = self._downstream_head(1, [tok.float() for tok in dec1], shape1, attention_masks=view1_attention_masks, max_instance_ids=view1_max_ids)
            res2, res2_masked, res2_attention_masks = self._downstream_head(2, [tok.float() for tok in dec2], shape2, attention_masks=view2_attention_masks, max_instance_ids=view2_max_ids)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        res2_masked['pts3d_in_other_view'] = res2_masked.pop('pts3d')  # predict view2's pts3d in view1's frame
        # print("res", res1['pts3d'].shape, res2['pts3d_in_other_view'].shape)
        res1_masked['attention_masks'] = res1_attention_masks
        res2_masked['attention_masks'] = res2_attention_masks
        if self.training:
            return res1, res2, res1_masked, res2_masked
        else:
            return res1, res2
        


class ObjectAwareDepthV2 (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 num_bins=5,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        self.num_bins = num_bins

        # dust3r specific initialization
        # Event RGB encoder convolutional to 4,H,W -> 3,H,W
        self.event_conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_bins, 32, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1), torch.nn.GELU(),   
        )
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), object_aware=True)
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), object_aware=True)
        # magic wrapper
        self.head1 = transpose_to_landscape_ob_aware(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape_ob_aware(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        ev1 = view1['event']#.unsqueeze(1)  # B,1,H,W
        ev2 = view2['event']#.unsqueeze(1)  # B,1,H,W

        # img1 = view1['img']  # B,3,H,W
        # img2 = view2['img']  # B,3,H,W

        # stack event and image along channel dimension
        # img1 = torch.cat((ev1, img1), dim=1)  # B,4,H,W
        # img2 = torch.cat((ev2, img2), dim=1)  # B,4,H,W

        B = ev1.shape[0]

        # print("ev1 shape", ev1.shape)
        # print("ev2 shape", ev2.shape)

        # print(view1['img'].shape) 

        img1 = self.event_conv(ev1)  # B,3,H,W
        img2 = self.event_conv(ev2)  # B,3,H,W
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)
        
        # print(img1.shape, img2.shape, view1['img'].shape, view2['img'].shape)

        # while (True):
        #     continue

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape, attention_masks=None, max_instance_ids=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        out, out_mask, attention_masks = head(decout, img_shape, attention_masks=attention_masks, max_instance_ids=max_instance_ids)
        return out, out_mask, attention_masks
    
    def _run_motion_segmentation(self, event_tensor):
        """
        event_tensor: [B, 5, H, W] torch float in [0,1] (assumed)
        returns: instance_ids [B, H, W] int32 (0 = background, 1..K = objects)
        """

        activity_maps = torch.abs(event_tensor).sum(dim=1)  # [B, H, W]
        B, H, W = activity_maps.shape
        device = event_tensor.device
        attention_masks = []

        activity_maps = activity_maps.detach().cpu().numpy()

        for b in range(B):
            activity_map = activity_maps[b]  # [H, W]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            activity_map = cv2.morphologyEx(activity_map, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            activity_map = cv2.morphologyEx(activity_map, cv2.MORPH_CLOSE, kernel)
            activity_map = activity_map.astype(np.uint8)*255

            _, binary = cv2.threshold(
                activity_map.astype(np.uint8), 0, 255,
                 cv2.THRESH_BINARY
            )
            non_binary = cv2.bitwise_not(binary)

            # stack them as a two-channel mask
            activity_map = np.stack([non_binary, binary], axis=0)  # [2, H, W]
            attention_masks.append(activity_map)

        # morphological operations to clean up the activity maps

        # visualize for debugging
        # create a torch of attention masks
        attention_masks = torch.from_numpy(np.stack(attention_masks, axis=0)).to(device).bool()  # [B, 2, H, W]

        # save a png for debugging from attention_masks[0]
        # overlay = np.zeros((H, W, 3), dtype=np.float32)
        # color_map = plt.get_cmap('tab10')
        # for inst_id in range(0,2):  # only two masks: background and motion
        #     color = color_map(inst_id)[:3]  # RGB
        #     mask = attention_masks[0, inst_id].cpu().numpy()
        #     overlay[mask] = color
        # # blend with the grayscale event image
        # event_img = event_tensor[0].sum(dim=0).detach().cpu().clamp(0, 1).numpy()
        # event_img_gray = np.stack([event_img]*3, axis=-1)
        # blended = (0.5 * event_img_gray + 0.5 * overlay).clip(0, 1)
        # plt.imsave("motion_segmentation_overlay.png", blended)
        # print("Saved motion segmentation overlay to motion_segmentation_overlay.png")
        return attention_masks, None, None  # [B, 2, H, W]


    def forward(self, view1, view2):
        # encode the two images --> B,S,D

        # print("Batch size in forward:", view1['img'].shape[0])

        # === NEW: if no instance_ids are provided, and SAM3 is enabled, run SAM3 on RGB ===
        # Expect view['img'] as [B, 3, H, W] if you want this path
            # if 'instance_ids' not in view1 and 'img' in view1:
        view1_attention_masks, _, view1_max_ids = self._run_motion_segmentation(view1['event'])
        # if 'instance_ids' not in view2 and 'img' in view2:
        view2_attention_masks, _, view2_max_ids= self._run_motion_segmentation(view2['event'])
        # === END NEW ===
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1, res1_masked, res1_attention_masks = self._downstream_head(1, [tok.float() for tok in dec1], shape1, attention_masks=view1_attention_masks, max_instance_ids=view1_max_ids)
            res2, res2_masked, res2_attention_masks = self._downstream_head(2, [tok.float() for tok in dec2], shape2, attention_masks=view2_attention_masks, max_instance_ids=view2_max_ids)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        res2_masked['pts3d_in_other_view'] = res2_masked.pop('pts3d')  # predict view2's pts3d in view1's frame
        # print("res", res1['pts3d'].shape, res2['pts3d_in_other_view'].shape)
        res1_masked['attention_masks'] = res1_attention_masks
        res2_masked['attention_masks'] = res2_attention_masks
        if self.training:
            return res1, res2, res1_masked, res2_masked
        else:
            return res1, res2