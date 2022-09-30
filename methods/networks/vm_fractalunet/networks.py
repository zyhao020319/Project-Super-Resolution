import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from methods.networks.voxelmorph.networks import Unet
from methods.networks.voxelmorph import layers


class FractalUnet(nn.Module):
    def __init__(self,
                 inshape=[240, 240],
                 infeats=2,
                 nb_features=16,
                 nb_levels=4,
                 max_pool=2,
                 feat_mult=2,
                 nb_conv_per_level=2,
                 half_res=False):
        super(FractalUnet, self).__init__()
        self.branches = nn.ModuleList(modules=[
            Unet(
                inshape=inshape, 
                infeats=infeats, 
                nb_features=nb_features, 
                nb_levels=2, 
                max_pool=max_pool, 
                feat_mult=feat_mult, 
                nb_conv_per_level=nb_conv_per_level
                )
            for _ in range(4)
        ])
        self.trunck = Unet(
            inshape=inshape,
            infeats=nb_features,
            nb_features=nb_features * 2,
            nb_levels=nb_levels,
            feat_mult=feat_mult,
            max_pool=max_pool,
            nb_conv_per_level=nb_conv_per_level
        )
        
    def forward(self, batch_mmis : torch.Tensor, batch_mfis : torch.Tensor, batch_are_existing : torch.Tensor):
        # batch_mmis : batch multimodal moving images -> [batch_size, 4, 240, 240]
        # batch_mfis : batch multimodal fixed images  -> [batch_size, 4, 240, 240]
        # batch_are_existing : 布尔数组                -> [batch_size, 4]
        batch_fused_feats = list()
        for mmis, mfis, are_existing, branch_unet in zip(batch_mmis, batch_mfis, batch_are_existing, self.branches):
            fused_feats = list()
            for moving_image, fixed_image, is_selected in zip(mmis, mfis, are_existing):
                if is_selected:
                    moving_image = moving_image.unsqueeze(0).unsqueeze(0)
                    fixed_image  = fixed_image.unsqueeze(0).unsqueeze(0)
                    x = torch.cat([moving_image, fixed_image], dim=1)
                    feat = branch_unet.forward(x)
                    fused_feats.append(feat)
            fused_feats = torch.max(torch.cat(fused_feats), dim=0, keepdim=True).values
            batch_fused_feats.append(fused_feats)
        batch_fused_feats = torch.cat(batch_fused_feats)
        
        batch_flow_logits = self.trunck.forward(batch_fused_feats)
        return batch_flow_logits


class VxmDenseFractalUnet(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super(VxmDenseFractalUnet, self).__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # self.branches = nn.ModuleList([
        #     Unet(
        #         inshape,
        #         infeats=(src_feats + trg_feats),
        #         nb_features=16,
        #         nb_levels=2,
        #         feat_mult=2,
        #         nb_conv_per_level=2,
        #         half_res=False,
        #     ) for _ in range(4)
        # ])
        
        # configure core unet model
        self.unet_model = FractalUnet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.trunck.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, are_selected, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''
        # concatenate inputs and propagate unet
        # x = torch.cat([source, target], dim=1)
        x = self.unet_model.forward(source, target, are_selected)
        print(x.shape)
        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out
