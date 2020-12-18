"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock

#def if_all_zero(tensor):
#    if torch.sum(tensor) == 0:
#        return 0
#    else:
#        return 1

def if_all_zero(tensor):
    b, _, _, _ = tensor.size()
    # print('b', b)
    index = torch.Tensor(b).fill_(1)
    # print('index size', index.size())
    # print('before index', index)
    for i in range(b):
        # print('i', i)
        if torch.sum(tensor[i:i+1, :, :, :]) == 0:
            index[i:i+1] = 0
    # print('after index', index)
    return index

class LGGANGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf  # of gen filters in first conv layer

        self.sw, self.sh = self.compute_latent_vector_size(opt)
        # print(self.sw, self.sh) 8, 4

        if opt.use_vae:  # False
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)  # print(self.opt.semantic_nc) # 36

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':  # opt.num_upsampling_layers: more
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        # local branch
        self.conv1 = nn.Conv2d(151, 64, 7, 1, 0) # change
        self.conv1_norm = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv4_norm = nn.InstanceNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 3, 2, 1)
        self.conv5_norm = nn.InstanceNorm2d(1024)

        self.resnet_blocks1 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks1.weight_init(0, 0.02)
        self.resnet_blocks2 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks2.weight_init(0, 0.02)
        self.resnet_blocks3 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks3.weight_init(0, 0.02)
        self.resnet_blocks4 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks4.weight_init(0, 0.02)
        self.resnet_blocks5 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks5.weight_init(0, 0.02)
        self.resnet_blocks6 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks6.weight_init(0, 0.02)
        self.resnet_blocks7 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks7.weight_init(0, 0.02)
        self.resnet_blocks8 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks8.weight_init(0, 0.02)
        self.resnet_blocks9 = resnet_block(256, 3, 1, 1)
        self.resnet_blocks9.weight_init(0, 0.02)

        self.deconv3_local = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_local = nn.InstanceNorm2d(128)
        self.deconv4_local = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_local = nn.InstanceNorm2d(64)

        self.deconv9 = nn.Conv2d(3*52, 3, 3, 1, 1)

        self.deconv5_0 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_1 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_2 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_3 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_4 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_5 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_6 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_7 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_8 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_9 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_10 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_11 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_12 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_13 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_14 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_15 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_16 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_17 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_18 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_19 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_20 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_21 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_22 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_23 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_24 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_25 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_26 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_27 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_28 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_29 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_30 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_31 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_32 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_33 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_34 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_35 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_36 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_37 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_38 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_39 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_40 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_41 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_42 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_43 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_44 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_45 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_46 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_47 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_48 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_49 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_50 = nn.Conv2d(64, 3, 7, 1, 0)
        self.deconv5_51 = nn.Conv2d(64, 3, 7, 1, 0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(64*256 * 512, 512)
        self.fc2 = nn.Linear(64, 51)

        self.deconv3_attention = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_attention = nn.InstanceNorm2d(128)
        self.deconv4_attention = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_attention = nn.InstanceNorm2d(64)
        self.deconv5_attention = nn.Conv2d(64, 2, 1, 1, 0)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        # global branch
        seg = input  #
        # print(input.size()) [1, 151, 256, 256]

        if self.opt.use_vae:  # use_vae: False
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            # x = F.interpolate(seg, size=(self.sh, self.sw))  # print(x.size()) [1, 36, 4, 8]
            # x = self.fc(x)  # print(x.size()) [1, 1024, 4, 8]

            x = F.pad(seg, (3, 3, 3, 3), 'reflect')  # print(x.size()) [1, 3, 262, 518]
            x = F.relu(self.conv1_norm(self.conv1(x)))  # print(x.size()) [1, 64, 256, 512]
            x = F.relu(self.conv2_norm(self.conv2(x)))  # print(x.size()) [1, 128, 128, 256]
            x_encode = F.relu(self.conv3_norm(self.conv3(x)))  # print(x.size()) [1, 256, 64, 128]
            x = F.relu(self.conv4_norm(self.conv4(x_encode)))  # print(x.size()) [1, 512, 32, 64]
            x = F.relu(self.conv5_norm(self.conv5(x)))  # print(x.size()) [1, 1024, 16, 32]
            x = F.interpolate(x, size=(self.sh, self.sw))  # print(x.size()) [1, 1024, 4, 8]

        x = self.head_0(x, seg)  # print(x.size()) [1, 1024, 4, 8] seg [1, 36, 256, 512]

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':  # num_upsampling_layers: more
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        result_global = F.tanh(x)

        ############################## local branch ##############################
        label_0_50 = input[:, 0: 51, :, :]
        # print(label_0_50.size()) [1, 51, 256, 256]
        label_51_151 = input[:, 51: 151, :, :]
        # print(label_51_151.size()) [1, 100, 256, 256]
        label_52 = torch.sum(label_51_151, dim=1, keepdim=True)
        # print(torch.max(label_52))

        label = torch.cat((label_0_50, label_52), 1)
        # print(label.size()) [1, 52, 256, 256]
        for i in range(52):
            globals()['label_' + str(i)] = label[:, i:i + 1, :, :]
            globals()['label_3_' + str(i)] = label[:, i:i + 1, :, :].repeat(1, 3, 1, 1)
            globals()['label_64_' + str(i)] = label[:, i:i + 1, :, :].repeat(1, 64, 1, 1)

        # print(x_encode.size()) [1, 256, 64, 64]
        x = self.resnet_blocks1(x_encode)
        x = self.resnet_blocks2(x)
        x = self.resnet_blocks3(x)
        x = self.resnet_blocks4(x)
        x = self.resnet_blocks5(x)
        x = self.resnet_blocks6(x)
        x = self.resnet_blocks7(x)
        x = self.resnet_blocks8(x) # print(x.size()) [1, 256, 64, 64]
        middle_x = self.resnet_blocks9(x) # print(middle_x.size()) [1, 256, 64, 128]
        x_local = F.relu(self.deconv3_norm_local(self.deconv3_local(middle_x))) # print(x_local.size()) [1, 128, 128, 256]
        x_feature_local = F.relu(self.deconv4_norm_local(self.deconv4_local(x_local))) # print(x_feature_local.size())

        # print(x_feature_local.size()) [1, 64, 256, 256]

        feature_0 = x_feature_local.cuda() * label_64_0.cuda()
        feature_1 = x_feature_local.cuda() * label_64_1.cuda()
        feature_2 = x_feature_local.cuda() * label_64_2.cuda()
        feature_3 = x_feature_local.cuda() * label_64_3.cuda()
        feature_4 = x_feature_local.cuda() * label_64_4.cuda()
        feature_5 = x_feature_local.cuda() * label_64_5.cuda()
        feature_6 = x_feature_local.cuda() * label_64_6.cuda()
        feature_7 = x_feature_local.cuda() * label_64_7.cuda()
        feature_8 = x_feature_local.cuda() * label_64_8.cuda()
        feature_9 = x_feature_local.cuda() * label_64_9.cuda()
        feature_10 = x_feature_local.cuda() * label_64_10.cuda()
        feature_11 = x_feature_local.cuda() * label_64_11.cuda()
        feature_12 = x_feature_local.cuda() * label_64_12.cuda()
        feature_13 = x_feature_local.cuda() * label_64_13.cuda()
        feature_14 = x_feature_local.cuda() * label_64_14.cuda()
        feature_15 = x_feature_local.cuda() * label_64_15.cuda()
        feature_16 = x_feature_local.cuda() * label_64_16.cuda()
        feature_17 = x_feature_local.cuda() * label_64_17.cuda()
        feature_18 = x_feature_local.cuda() * label_64_18.cuda()
        feature_19 = x_feature_local.cuda() * label_64_19.cuda()
        feature_20 = x_feature_local.cuda() * label_64_20.cuda()
        feature_21 = x_feature_local.cuda() * label_64_21.cuda()
        feature_22 = x_feature_local.cuda() * label_64_22.cuda()
        feature_23 = x_feature_local.cuda() * label_64_23.cuda()
        feature_24 = x_feature_local.cuda() * label_64_24.cuda()
        feature_25 = x_feature_local.cuda() * label_64_25.cuda()
        feature_26 = x_feature_local.cuda() * label_64_26.cuda()
        feature_27 = x_feature_local.cuda() * label_64_27.cuda()
        feature_28 = x_feature_local.cuda() * label_64_28.cuda()
        feature_29 = x_feature_local.cuda() * label_64_29.cuda()
        feature_30 = x_feature_local.cuda() * label_64_30.cuda()
        feature_31 = x_feature_local.cuda() * label_64_31.cuda()
        feature_32 = x_feature_local.cuda() * label_64_32.cuda()
        feature_33 = x_feature_local.cuda() * label_64_33.cuda()
        feature_34 = x_feature_local.cuda() * label_64_34.cuda()
        feature_35 = x_feature_local.cuda() * label_64_35.cuda()
        feature_36 = x_feature_local.cuda() * label_64_36.cuda()
        feature_37 = x_feature_local.cuda() * label_64_37.cuda()
        feature_38 = x_feature_local.cuda() * label_64_38.cuda()
        feature_39 = x_feature_local.cuda() * label_64_39.cuda()
        feature_40 = x_feature_local.cuda() * label_64_40.cuda()
        feature_41 = x_feature_local.cuda() * label_64_41.cuda()
        feature_42 = x_feature_local.cuda() * label_64_42.cuda()
        feature_43 = x_feature_local.cuda() * label_64_43.cuda()
        feature_44 = x_feature_local.cuda() * label_64_44.cuda()
        feature_45 = x_feature_local.cuda() * label_64_45.cuda()
        feature_46 = x_feature_local.cuda() * label_64_46.cuda()
        feature_47 = x_feature_local.cuda() * label_64_47.cuda()
        feature_48 = x_feature_local.cuda() * label_64_48.cuda()
        feature_49 = x_feature_local.cuda() * label_64_49.cuda()
        feature_50 = x_feature_local.cuda() * label_64_50.cuda()
        feature_51 = x_feature_local.cuda() * label_64_51.cuda()
        # print('before feature:', feature_0.size())

        # print('after feature:', feature_0.size())
        # [1, 64, 256, 512]
        feature_combine= torch.cat((feature_0,feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11,
                                    feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18, feature_19,feature_20,feature_21,
                                    feature_22,feature_23, feature_24, feature_25, feature_26,feature_27,feature_28,feature_29,feature_30,feature_31,
                                    feature_32,feature_33,feature_34, feature_35, feature_36, feature_37, feature_38, feature_39, feature_40, feature_41,
                                    feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49, feature_50), 0)
        # print(feature_combine.size())
        # [52, 64, 256, 256]
        feature_combine = self.avgpool(feature_combine)
        # print(feature_combine.size())
        # [52, 64 , 1, 1]
        feature_combine_fc = torch.flatten(feature_combine, 1)
        # print(feature_combine_fc.size())
        # [52, 64]
        feature_score = self.fc2(feature_combine_fc)
        # print(feature_score.size()) [35, 35]
        #target= torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,
         #                     34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49])
        # print(target)
        b, _, _, _ = feature_34.size()
        # print(feature_score.size()) [35, 35]
        target_label_0 = torch.Tensor(b).fill_(0)
        target_label_1 = torch.Tensor(b).fill_(1)
        target_label_2 = torch.Tensor(b).fill_(2)
        target_label_3 = torch.Tensor(b).fill_(3)
        target_label_4 = torch.Tensor(b).fill_(4)
        target_label_5 = torch.Tensor(b).fill_(5)
        target_label_6 = torch.Tensor(b).fill_(6)
        target_label_7 = torch.Tensor(b).fill_(7)
        target_label_8 = torch.Tensor(b).fill_(8)
        target_label_9 = torch.Tensor(b).fill_(9)
        target_label_10 = torch.Tensor(b).fill_(10)
        target_label_11 = torch.Tensor(b).fill_(11)
        target_label_12 = torch.Tensor(b).fill_(12)
        target_label_13 = torch.Tensor(b).fill_(13)
        target_label_14 = torch.Tensor(b).fill_(14)
        target_label_15 = torch.Tensor(b).fill_(15)
        target_label_16 = torch.Tensor(b).fill_(16)
        target_label_17 = torch.Tensor(b).fill_(17)
        target_label_18 = torch.Tensor(b).fill_(18)
        target_label_19 = torch.Tensor(b).fill_(19)
        target_label_20 = torch.Tensor(b).fill_(20)
        target_label_21 = torch.Tensor(b).fill_(21)
        target_label_22 = torch.Tensor(b).fill_(22)
        target_label_23 = torch.Tensor(b).fill_(23)
        target_label_24 = torch.Tensor(b).fill_(24)
        target_label_25 = torch.Tensor(b).fill_(25)
        target_label_26 = torch.Tensor(b).fill_(26)
        target_label_27 = torch.Tensor(b).fill_(27)
        target_label_28 = torch.Tensor(b).fill_(28)
        target_label_29 = torch.Tensor(b).fill_(29)
        target_label_30 = torch.Tensor(b).fill_(30)
        target_label_31 = torch.Tensor(b).fill_(31)
        target_label_32 = torch.Tensor(b).fill_(32)
        target_label_33 = torch.Tensor(b).fill_(33)
        target_label_34 = torch.Tensor(b).fill_(34)

        target_label_35 = torch.Tensor(b).fill_(35)
        target_label_36 = torch.Tensor(b).fill_(36)
        target_label_37 = torch.Tensor(b).fill_(37)
        target_label_38 = torch.Tensor(b).fill_(38)
        target_label_39 = torch.Tensor(b).fill_(39)
        target_label_40 = torch.Tensor(b).fill_(40)
        target_label_41 = torch.Tensor(b).fill_(41)
        target_label_42 = torch.Tensor(b).fill_(42)
        target_label_43 = torch.Tensor(b).fill_(43)
        target_label_44 = torch.Tensor(b).fill_(44)
        target_label_45 = torch.Tensor(b).fill_(45)
        target_label_46 = torch.Tensor(b).fill_(46)
        target_label_47 = torch.Tensor(b).fill_(47)
        target_label_48 = torch.Tensor(b).fill_(48)
        target_label_49 = torch.Tensor(b).fill_(49)
        target_label_50 = torch.Tensor(b).fill_(50)

        target= torch.cat((target_label_0, target_label_1, target_label_2, target_label_3, target_label_4, target_label_5, target_label_6, target_label_7,
                           target_label_8, target_label_9, target_label_10, target_label_11, target_label_12, target_label_13, target_label_14,
                           target_label_15, target_label_16, target_label_17, target_label_18, target_label_19,target_label_20,target_label_21,
                           target_label_22,target_label_23, target_label_24, target_label_25, target_label_26,target_label_27,target_label_28,
                           target_label_29,target_label_30,target_label_31,target_label_32,target_label_33,target_label_34,target_label_35,target_label_36,target_label_37,target_label_38,target_label_39,target_label_40,target_label_41,target_label_42,target_label_43,target_label_44,target_label_45,target_label_46,target_label_47,target_label_48,target_label_49,target_label_50), 0)
        target=target.long()        

        # print(label_0)
        valid_index = torch.cat((if_all_zero(label_0),if_all_zero(label_1), if_all_zero(label_2),if_all_zero(label_3),if_all_zero(label_4), if_all_zero(label_5),
                                    if_all_zero(label_6), if_all_zero(label_7), if_all_zero(label_8),if_all_zero(label_9),if_all_zero(label_10),if_all_zero(label_11),
                                    if_all_zero(label_12),if_all_zero(label_13),if_all_zero(label_14),if_all_zero(label_15),if_all_zero(label_16),if_all_zero(label_17),
                                    if_all_zero(label_18),if_all_zero(label_19),if_all_zero(label_20),if_all_zero(label_21),if_all_zero(label_22),if_all_zero(label_23),
                                    if_all_zero(label_24),if_all_zero(label_25),if_all_zero(label_26),if_all_zero(label_27),if_all_zero(label_28),if_all_zero(label_29),
                                    if_all_zero(label_30),if_all_zero(label_31),if_all_zero(label_32),if_all_zero(label_33),if_all_zero(label_34),if_all_zero(label_35),
                                    if_all_zero(label_36),if_all_zero(label_37),if_all_zero(label_38),if_all_zero(label_39),if_all_zero(label_40),if_all_zero(label_41),
                                    if_all_zero(label_42),if_all_zero(label_43),if_all_zero(label_44),if_all_zero(label_45),if_all_zero(label_46),if_all_zero(label_47),
                                    if_all_zero(label_48),if_all_zero(label_49),if_all_zero(label_50)),0)
        # print(valid_index)


        # for i in range(self.opt.label_nc):
        #     globals()['feature_' + str(i)] = F.pad(eval('feature_%d'% (i)), (3, 3, 3, 3), 'reflect').cuda() # print(label_1.size())  [1, 64, 262, 518]

        feature_0 = F.pad(feature_0, (3, 3, 3, 3), 'reflect')
        feature_1 = F.pad(feature_1, (3, 3, 3, 3), 'reflect')
        feature_2 = F.pad(feature_2, (3, 3, 3, 3), 'reflect')
        feature_3 = F.pad(feature_3, (3, 3, 3, 3), 'reflect')
        feature_4 = F.pad(feature_4, (3, 3, 3, 3), 'reflect')
        feature_5 = F.pad(feature_5, (3, 3, 3, 3), 'reflect')
        feature_6 = F.pad(feature_6, (3, 3, 3, 3), 'reflect')
        feature_7 = F.pad(feature_7, (3, 3, 3, 3), 'reflect')
        feature_8 = F.pad(feature_8, (3, 3, 3, 3), 'reflect')
        feature_9 = F.pad(feature_9, (3, 3, 3, 3), 'reflect')
        feature_10 = F.pad(feature_10, (3, 3, 3, 3), 'reflect')
        feature_11 = F.pad(feature_11, (3, 3, 3, 3), 'reflect')
        feature_12 = F.pad(feature_12, (3, 3, 3, 3), 'reflect')
        feature_13 = F.pad(feature_13, (3, 3, 3, 3), 'reflect')
        feature_14 = F.pad(feature_14, (3, 3, 3, 3), 'reflect')
        feature_15 = F.pad(feature_15, (3, 3, 3, 3), 'reflect')
        feature_16 = F.pad(feature_16, (3, 3, 3, 3), 'reflect')
        feature_17 = F.pad(feature_17, (3, 3, 3, 3), 'reflect')
        feature_18 = F.pad(feature_18, (3, 3, 3, 3), 'reflect')
        feature_19 = F.pad(feature_19, (3, 3, 3, 3), 'reflect')
        feature_20 = F.pad(feature_20, (3, 3, 3, 3), 'reflect')
        feature_21 = F.pad(feature_21, (3, 3, 3, 3), 'reflect')
        feature_22 = F.pad(feature_22, (3, 3, 3, 3), 'reflect')
        feature_23 = F.pad(feature_23, (3, 3, 3, 3), 'reflect')
        feature_24 = F.pad(feature_24, (3, 3, 3, 3), 'reflect')
        feature_25 = F.pad(feature_25, (3, 3, 3, 3), 'reflect')
        feature_26 = F.pad(feature_26, (3, 3, 3, 3), 'reflect')
        feature_27 = F.pad(feature_27, (3, 3, 3, 3), 'reflect')
        feature_28 = F.pad(feature_28, (3, 3, 3, 3), 'reflect')
        feature_29 = F.pad(feature_29, (3, 3, 3, 3), 'reflect')
        feature_30 = F.pad(feature_30, (3, 3, 3, 3), 'reflect')
        feature_31 = F.pad(feature_31, (3, 3, 3, 3), 'reflect')
        feature_32 = F.pad(feature_32, (3, 3, 3, 3), 'reflect')
        feature_33 = F.pad(feature_33, (3, 3, 3, 3), 'reflect')
        feature_34 = F.pad(feature_34, (3, 3, 3, 3), 'reflect')
        feature_35 = F.pad(feature_35, (3, 3, 3, 3), 'reflect')
        feature_36 = F.pad(feature_36, (3, 3, 3, 3), 'reflect')
        feature_37 = F.pad(feature_37, (3, 3, 3, 3), 'reflect')
        feature_38 = F.pad(feature_38, (3, 3, 3, 3), 'reflect')
        feature_39 = F.pad(feature_39, (3, 3, 3, 3), 'reflect')
        feature_40 = F.pad(feature_40, (3, 3, 3, 3), 'reflect')
        feature_41 = F.pad(feature_41, (3, 3, 3, 3), 'reflect')
        feature_42 = F.pad(feature_42, (3, 3, 3, 3), 'reflect')
        feature_43 = F.pad(feature_43, (3, 3, 3, 3), 'reflect')
        feature_44 = F.pad(feature_44, (3, 3, 3, 3), 'reflect')
        feature_45 = F.pad(feature_45, (3, 3, 3, 3), 'reflect')
        feature_46 = F.pad(feature_46, (3, 3, 3, 3), 'reflect')
        feature_47 = F.pad(feature_47, (3, 3, 3, 3), 'reflect')
        feature_48 = F.pad(feature_48, (3, 3, 3, 3), 'reflect')
        feature_49 = F.pad(feature_49, (3, 3, 3, 3), 'reflect')
        feature_50 = F.pad(feature_50, (3, 3, 3, 3), 'reflect')
        feature_51 = F.pad(feature_51, (3, 3, 3, 3), 'reflect')


        result_0 = torch.tanh(self.deconv5_0(feature_0.cuda())) # print(result_1.size()) [1, 3, 256, 512]
        # print(result_0.size())
        result_1 = torch.tanh(self.deconv5_1(feature_1.cuda()))
        result_2 = torch.tanh(self.deconv5_2(feature_2.cuda()))
        result_3 = torch.tanh(self.deconv5_3(feature_3.cuda()))
        result_4 = torch.tanh(self.deconv5_4(feature_4.cuda()))
        result_5 = torch.tanh(self.deconv5_5(feature_5.cuda()))
        result_6 = torch.tanh(self.deconv5_6(feature_6.cuda()))
        result_7 = torch.tanh(self.deconv5_7(feature_7.cuda()))
        result_8 = torch.tanh(self.deconv5_8(feature_8.cuda()))
        result_9 = torch.tanh(self.deconv5_9(feature_9.cuda()))
        result_10 = torch.tanh(self.deconv5_10(feature_10.cuda()))
        result_11 = torch.tanh(self.deconv5_11(feature_11.cuda()))
        result_12 = torch.tanh(self.deconv5_12(feature_12.cuda()))
        result_13 = torch.tanh(self.deconv5_13(feature_13.cuda()))
        result_14 = torch.tanh(self.deconv5_14(feature_14.cuda()))
        result_15 = torch.tanh(self.deconv5_15(feature_15.cuda()))
        result_16 = torch.tanh(self.deconv5_16(feature_16.cuda()))
        result_17 = torch.tanh(self.deconv5_17(feature_17.cuda()))
        result_18 = torch.tanh(self.deconv5_18(feature_18.cuda()))
        result_19 = torch.tanh(self.deconv5_19(feature_19.cuda()))
        result_20 = torch.tanh(self.deconv5_20(feature_20.cuda()))
        result_21 = torch.tanh(self.deconv5_21(feature_21.cuda()))
        result_22 = torch.tanh(self.deconv5_22(feature_22.cuda()))
        result_23 = torch.tanh(self.deconv5_23(feature_23.cuda()))
        result_24 = torch.tanh(self.deconv5_24(feature_24.cuda()))
        result_25 = torch.tanh(self.deconv5_25(feature_25.cuda()))
        result_26 = torch.tanh(self.deconv5_26(feature_26.cuda()))
        result_27 = torch.tanh(self.deconv5_27(feature_27.cuda()))
        result_28 = torch.tanh(self.deconv5_28(feature_28.cuda()))
        result_29 = torch.tanh(self.deconv5_29(feature_29.cuda()))
        result_30 = torch.tanh(self.deconv5_30(feature_30.cuda()))
        result_31 = torch.tanh(self.deconv5_31(feature_31.cuda()))
        result_32 = torch.tanh(self.deconv5_32(feature_32.cuda()))
        result_33 = torch.tanh(self.deconv5_33(feature_33.cuda()))
        result_34 = torch.tanh(self.deconv5_34(feature_34.cuda()))
        result_35 = torch.tanh(self.deconv5_35(feature_35.cuda()))
        result_36 = torch.tanh(self.deconv5_36(feature_36.cuda()))
        result_37 = torch.tanh(self.deconv5_37(feature_37.cuda()))
        result_38 = torch.tanh(self.deconv5_38(feature_38.cuda()))
        result_39 = torch.tanh(self.deconv5_39(feature_39.cuda()))
        result_40 = torch.tanh(self.deconv5_40(feature_40.cuda()))
        result_41 = torch.tanh(self.deconv5_41(feature_41.cuda()))
        result_42 = torch.tanh(self.deconv5_42(feature_42.cuda()))
        result_43 = torch.tanh(self.deconv5_43(feature_43.cuda()))
        result_44 = torch.tanh(self.deconv5_44(feature_44.cuda()))
        result_45 = torch.tanh(self.deconv5_45(feature_45.cuda()))
        result_46 = torch.tanh(self.deconv5_46(feature_46.cuda()))
        result_47 = torch.tanh(self.deconv5_47(feature_47.cuda()))
        result_48 = torch.tanh(self.deconv5_48(feature_48.cuda()))
        result_49 = torch.tanh(self.deconv5_49(feature_49.cuda()))
        result_50 = torch.tanh(self.deconv5_50(feature_50.cuda()))
        result_51 = torch.tanh(self.deconv5_51(feature_51.cuda()))


        # result_local = result_0 + result_1 + result_2 + result_3 + result_4 + result_5 + result_6 + result_7 + result_8 + result_9 + result_10 + \
        #         result_11 + result_12 + result_13 + result_14 + result_15 + result_16 + result_17 + result_18 + result_19 + result_20 + \
        #         result_21 + result_22 + result_23 + result_24 + result_25 + result_26 + result_27 + result_28 + result_29 + result_30 + \
        #                result_31 + result_32 + result_33 + result_34

        combine_local = torch.cat((result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
                                   result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
                                   result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
                                   result_31 , result_32 , result_33 , result_34, result_35, result_36, result_37, result_38, result_39, result_40,
                                   result_41, result_42, result_43, result_44, result_45, result_46, result_47, result_48, result_49, result_50, result_51), 1)
        result_local = torch.tanh(self.deconv9(combine_local))

        # print(result_local.size()) [1, 3, 262, 518]
        # print(result_global.size()) [1, 3, 256, 512]
        # final = (result_global + result_local) * 0.5

        # x_encode [1 ,256, 64, 128]
        x_attention = F.relu(self.deconv3_norm_attention(self.deconv3_attention(x_encode)))
        # print(x_attention.size()) [1, 128, 128, 256]
        x_attention = F.relu(self.deconv4_norm_attention(self.deconv4_attention(x_attention)))
        # print(x_attention.size()) [1, 64, 256, 512]
        result_attention = self.deconv5_attention(x_attention)
        # print(result_attention.size()) [1, 2, 256, 512]
        softmax_ = torch.nn.Softmax(dim=1)
        result_attention = softmax_(result_attention)

        attention_local = result_attention[:, 0:1, :, :]
        attention_global = result_attention[:, 1:2, :, :]

        attention_local = attention_local.repeat(1, 3, 1, 1)
        attention_global = attention_global.repeat(1, 3, 1, 1)

        final = attention_local * result_local + attention_global * result_global

        # attention_global_v = (attention_global - 0.5)/0.5 # for visualization
        # attention_local_v =  (attention_local - 0.5)/0.5 # for visualization
        # final = (result_global + result_local) * 0.5


        return final, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
               label_3_32, label_3_33, label_3_34, label_3_35, label_3_36,label_3_37,label_3_38,label_3_39,label_3_40,label_3_41,label_3_42,label_3_43, \
               label_3_44,label_3_45,label_3_46,label_3_47,label_3_48,label_3_49,label_3_50,label_3_51,\
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
               result_31 , result_32 , result_33 , result_34, result_35 , result_36 , result_37 , result_38,result_39 , result_40 , result_41 , \
               result_42,result_43,result_44,result_45,result_46,result_47,result_48,result_49,result_50,result_51,\
               feature_score, target.cuda(), valid_index.float().cuda(), attention_global, attention_local


# resnet block with reflect padding
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9,
                            help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
