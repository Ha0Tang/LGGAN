"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCE = torch.nn.CrossEntropyLoss(reduction='none')
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data) # print(input_semantics.size()) [1, 36, 256, 512] print(real_image.size()) [1, 3, 256, 512]
        # print(label_map.size()) [1, 1, 256, 512]

        if mode == 'generator':
            g_loss, generated, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
               label_3_32, label_3_33, label_3_34, label_3_35, label_3_36,label_3_37,label_3_38,label_3_39,label_3_40,label_3_41,label_3_42,label_3_43, \
               label_3_44,label_3_45,label_3_46,label_3_47,label_3_48,label_3_49,label_3_50,label_3_51,\
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
               result_31 , result_32 , result_33 , result_34, result_35 , result_36 , result_37 , result_38,result_39 , result_40 , result_41 , \
               result_42,result_43,result_44,result_45,result_46,result_47,result_48,result_49,result_50,result_51, feature_score, target, index, attention_global, attention_local = self.compute_generator_loss(input_semantics, real_image)
            return g_loss, generated, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
               label_3_32, label_3_33, label_3_34, label_3_35, label_3_36,label_3_37,label_3_38,label_3_39,label_3_40,label_3_41,label_3_42,label_3_43, \
               label_3_44,label_3_45,label_3_46,label_3_47,label_3_48,label_3_49,label_3_50,label_3_51,\
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
               result_31 , result_32 , result_33 , result_34, result_35 , result_36 , result_37 , result_38,result_39 , result_40 , result_41 , \
               result_42,result_43,result_44,result_45,result_46,result_47,result_48,result_49,result_50,result_51,feature_score, target, index, attention_global, attention_local
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
               label_3_32, label_3_33, label_3_34, label_3_35, label_3_36,label_3_37,label_3_38,label_3_39,label_3_40,label_3_41,label_3_42,label_3_43, \
               label_3_44,label_3_45,label_3_46,label_3_47,label_3_48,label_3_49,label_3_50,label_3_51,\
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
               result_31 , result_32 , result_33 , result_34, result_35 , result_36 , result_37 , result_38,result_39 , result_40 , result_41 , \
               result_42,result_43,result_44,result_45,result_46,result_47,result_48,result_49,result_50,result_51,feature_score, target, index,  attention_global, attention_local, _ = self.generate_fake(input_semantics, real_image)
            return fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
               label_3_32, label_3_33, label_3_34, label_3_35, label_3_36,label_3_37,label_3_38,label_3_39,label_3_40,label_3_41,label_3_42,label_3_43, \
               label_3_44,label_3_45,label_3_46,label_3_47,label_3_48,label_3_49,label_3_50,label_3_51,\
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
               result_31 , result_32 , result_33 , result_34, result_35 , result_36 , result_37 , result_38,result_39 , result_40 , result_41 , \
               result_42,result_43,result_44,result_45,result_46,result_47,result_48,result_49,result_50,result_51,feature_score, target, index,  attention_global, attention_local
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        label_map = data['label'] # print(label_map.size()) [1, 1, 256, 512]
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_() # print(input_label.size()) [1, 35, 256, 512]
        input_semantics = input_label.scatter_(1, label_map, 1.0) # print(input_semantics.size()) [1, 35, 256, 512]

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
               label_3_32, label_3_33, label_3_34, label_3_35, label_3_36,label_3_37,label_3_38,label_3_39,label_3_40,label_3_41,label_3_42,label_3_43, \
               label_3_44,label_3_45,label_3_46,label_3_47,label_3_48,label_3_49,label_3_50,label_3_51,\
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
               result_31 , result_32 , result_33 , result_34, result_35 , result_36 , result_37 , result_38,result_39 , result_40 , result_41 , \
               result_42,result_43,result_44,result_45,result_46,result_47,result_48,result_49,result_50,result_51,feature_score, target, index, attention_global, attention_local, KLD_loss = self.generate_fake(input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake_generated, pred_real_generated = self.discriminate(input_semantics, fake_image, real_image)
        pred_fake_global, pred_real_global = self.discriminate(input_semantics, result_global, real_image)
        pred_fake_local, pred_real_local = self.discriminate(input_semantics, result_local, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake_generated, True, for_discriminator=False) + \
                          self.criterionGAN(pred_fake_global, True, for_discriminator=False) + \
                          self.criterionGAN(pred_fake_local, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake_generated) # print(num_D) 2

            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator, last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake_generated[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(pred_fake_generated[i][j], pred_real_generated[i][j].detach()) + \
                                      self.criterionFeat(pred_fake_global[i][j], pred_real_global[i][j].detach()) + \
                                      self.criterionFeat(pred_fake_local[i][j], pred_real_local[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg + \
                              self.criterionVGG(result_global, real_image) * self.opt.lambda_vgg + \
                              self.criterionVGG(result_local, real_image) * self.opt.lambda_vgg

        if not self.opt.no_l1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) * self.opt.lambda_l1 + \
                              self.criterionL1(result_global, real_image) * self.opt.lambda_l1 + \
                              self.criterionL1(result_local, real_image) * self.opt.lambda_l1

        if not self.opt.no_class_loss:
            # print(self.criterionCE(feature_score, target))
            # print(index.type()) torch.cuda.LongTensor
            # print(self.criterionCE(feature_score, target) * index)
            # print(torch.sum(self.criterionCE(feature_score, target) * index))
            # print(torch.sum(index))
            # print(torch.sum(self.criterionCE(feature_score, target) * index)/torch.sum(index))
            G_losses['class'] = torch.sum(self.criterionCE(feature_score, target) * index) / torch.sum(index) * self.opt.lambda_class
        # print(self.criterionCE(feature_score, target).size())

        # TO DO: local pixel loss
        if not self.opt.no_l1_local_loss:
            real_image = real_image.cuda()
            real_0 = real_image * label_3_0.cuda()
            real_1 = real_image * label_3_1.cuda()
            real_2 = real_image * label_3_2.cuda()
            real_3 = real_image * label_3_3.cuda()
            real_4 = real_image * label_3_4.cuda()
            real_5 = real_image * label_3_5.cuda()
            real_6 = real_image * label_3_6.cuda()
            real_7 = real_image * label_3_7.cuda()
            real_8 = real_image * label_3_8.cuda()
            real_9 = real_image * label_3_9.cuda()
            real_10 = real_image * label_3_10.cuda()
            real_11 = real_image * label_3_11.cuda()
            real_12 = real_image * label_3_12.cuda()
            real_13 = real_image * label_3_13.cuda()
            real_14 = real_image * label_3_14.cuda()
            real_15 = real_image * label_3_15.cuda()
            real_16 = real_image * label_3_16.cuda()
            real_17 = real_image * label_3_17.cuda()
            real_18 = real_image * label_3_18.cuda()
            real_19 = real_image * label_3_19.cuda()
            real_20 = real_image * label_3_20.cuda()
            real_21 = real_image * label_3_21.cuda()
            real_22 = real_image * label_3_22.cuda()
            real_23 = real_image * label_3_23.cuda()
            real_24 = real_image * label_3_24.cuda()
            real_25 = real_image * label_3_25.cuda()
            real_26 = real_image * label_3_26.cuda()
            real_27 = real_image * label_3_27.cuda()
            real_28 = real_image * label_3_28.cuda()
            real_29 = real_image * label_3_29.cuda()
            real_30 = real_image * label_3_30.cuda()
            real_31 = real_image * label_3_31.cuda()
            real_32 = real_image * label_3_32.cuda()
            real_33 = real_image * label_3_33.cuda()
            real_34 = real_image * label_3_34.cuda()

            real_35 = real_image * label_3_35.cuda()
            real_36 = real_image * label_3_36.cuda()
            real_37 = real_image * label_3_37.cuda()
            real_38 = real_image * label_3_38.cuda()
            real_39 = real_image * label_3_39.cuda()
            real_40 = real_image * label_3_40.cuda()
            real_41 = real_image * label_3_41.cuda()
            real_42 = real_image * label_3_42.cuda()
            real_43 = real_image * label_3_43.cuda()
            real_44 = real_image * label_3_44.cuda()
            real_45 = real_image * label_3_45.cuda()
            real_46 = real_image * label_3_46.cuda()
            real_47 = real_image * label_3_47.cuda()
            real_48 = real_image * label_3_48.cuda()
            real_49 = real_image * label_3_49.cuda()
            real_50 = real_image * label_3_50.cuda()
            real_51 = real_image * label_3_51.cuda()

            G_losses['L1_Local'] = self.opt.lambda_l1 * ( self.criterionL1(result_0.cuda(), real_0) + \
                              self.criterionL1(result_1.cuda(), real_1) + self.criterionL1(result_2.cuda(), real_2) + \
                              self.criterionL1(result_3.cuda(), real_3) + self.criterionL1(result_4.cuda(), real_4) + \
                              self.criterionL1(result_5.cuda(), real_5) + self.criterionL1(result_6.cuda(), real_6) + \
                              self.criterionL1(result_7.cuda(), real_7) + self.criterionL1(result_8.cuda(), real_8) + \
                              self.criterionL1(result_9.cuda(), real_9) + self.criterionL1(result_10.cuda(), real_10) + \
                              self.criterionL1(result_11.cuda(), real_11) + self.criterionL1(result_12.cuda(), real_12) + \
                              self.criterionL1(result_13.cuda(), real_13) + self.criterionL1(result_14.cuda(), real_14) + \
                              self.criterionL1(result_15.cuda(), real_15) + self.criterionL1(result_16.cuda(), real_16) + \
                              self.criterionL1(result_17.cuda(), real_17) + self.criterionL1(result_18.cuda(), real_18) + \
                              self.criterionL1(result_19.cuda(), real_19) + self.criterionL1(result_20.cuda(), real_20) + \
                              self.criterionL1(result_21.cuda(), real_21) + self.criterionL1(result_22.cuda(), real_22) + \
                              self.criterionL1(result_23.cuda(), real_23) + self.criterionL1(result_24.cuda(), real_24) + \
                              self.criterionL1(result_25.cuda(), real_25) + self.criterionL1(result_26.cuda(), real_26) + \
                              self.criterionL1(result_27.cuda(), real_27) + self.criterionL1(result_28.cuda(), real_28) + \
                              self.criterionL1(result_29.cuda(), real_29) + self.criterionL1(result_30.cuda(), real_30) + \
                              self.criterionL1(result_31.cuda(), real_31) + self.criterionL1(result_32.cuda(), real_32) + \
                              self.criterionL1(result_33.cuda(), real_33) + self.criterionL1(result_34.cuda(), real_34) + \
                              self.criterionL1(result_35.cuda(), real_35) + self.criterionL1(result_36.cuda(), real_36) + \
                              self.criterionL1(result_37.cuda(), real_37) + self.criterionL1(result_38.cuda(), real_38) +\
                              self.criterionL1(result_39.cuda(), real_39) + self.criterionL1(result_40.cuda(), real_40)+\
                              self.criterionL1(result_41.cuda(), real_41) + self.criterionL1(result_42.cuda(), real_42)+\
                              self.criterionL1(result_43.cuda(), real_43) + self.criterionL1(result_44.cuda(), real_44)+\
                              self.criterionL1(result_45.cuda(), real_45) + self.criterionL1(result_46.cuda(), real_46)+\
                              self.criterionL1(result_47.cuda(), real_47) + self.criterionL1(result_48.cuda(), real_48)+\
                              self.criterionL1(result_49.cuda(), real_49) + self.criterionL1(result_50.cuda(), real_50)+\
                              self.criterionL1(result_51.cuda(), real_51))

        return G_losses, fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
               label_3_32, label_3_33, label_3_34, label_3_35, label_3_36,label_3_37,label_3_38,label_3_39,label_3_40,label_3_41,label_3_42,label_3_43, \
               label_3_44,label_3_45,label_3_46,label_3_47,label_3_48,label_3_49,label_3_50,label_3_51,\
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
               result_31 , result_32 , result_33 , result_34, result_35 , result_36 , result_37 , result_38,result_39 , result_40 , result_41 , \
               result_42,result_43,result_44,result_45,result_46,result_47,result_48,result_49,result_50,result_51,feature_score, target, index, attention_global, attention_local
        # return G_losses, fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
        #        label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
        #        label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
        #        label_3_32, label_3_33, label_3_34, result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
        #        result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
        #        result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
        #        result_31 , result_32 , result_33 , result_34,feature_score, target, index, attention_global, attention_local

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
               label_3_32, label_3_33, label_3_34, label_3_35, label_3_36,label_3_37,label_3_38,label_3_39,label_3_40,label_3_41,label_3_42,label_3_43, \
               label_3_44,label_3_45,label_3_46,label_3_47,label_3_48,label_3_49,label_3_50,label_3_51,\
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
               result_31 , result_32 , result_33 , result_34, result_35 , result_36 , result_37 , result_38,result_39 , result_40 , result_41 , \
               result_42,result_43,result_44,result_45,result_46,result_47,result_48,result_49,result_50,result_51,feature_score, target, valid_index, attention_global, attention_local, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake_generated, pred_real_generated = self.discriminate(input_semantics, fake_image, real_image)
        pred_fake_global, pred_real_global = self.discriminate(input_semantics, result_global, real_image)
        pred_fake_lcoal, pred_real_local = self.discriminate(input_semantics, result_local, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake_generated, False, for_discriminator=True) + \
                             self.criterionGAN(pred_fake_global, False, for_discriminator=True) + \
                             self.criterionGAN(pred_fake_lcoal, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real_generated, True, for_discriminator=True) + \
                             self.criterionGAN(pred_real_global, True, for_discriminator=True) + \
                             self.criterionGAN(pred_real_local, True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
               label_3_32, label_3_33, label_3_34, label_3_35, label_3_36,label_3_37,label_3_38,label_3_39,label_3_40,label_3_41,label_3_42,label_3_43, \
               label_3_44,label_3_45,label_3_46,label_3_47,label_3_48,label_3_49,label_3_50,label_3_51,\
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
               result_31 , result_32 , result_33 , result_34, result_35 , result_36 , result_37 , result_38,result_39 , result_40 , result_41 , \
               result_42,result_43,result_44,result_45,result_46,result_47,result_48,result_49,result_50,result_51, feature_score, target, index, attention_global, attention_local = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, label_3_29, label_3_30, label_3_31, \
               label_3_32, label_3_33, label_3_34, label_3_35, label_3_36,label_3_37,label_3_38,label_3_39,label_3_40,label_3_41,label_3_42,label_3_43, \
               label_3_44,label_3_45,label_3_46,label_3_47,label_3_48,label_3_49,label_3_50,label_3_51,\
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , result_29 , result_30, \
               result_31 , result_32 , result_33 , result_34, result_35 , result_36 , result_37 , result_38,result_39 , result_40 , result_41 , \
               result_42,result_43,result_44,result_45,result_46,result_47,result_48,result_49,result_50,result_51, feature_score, target,index, attention_global, attention_local, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
