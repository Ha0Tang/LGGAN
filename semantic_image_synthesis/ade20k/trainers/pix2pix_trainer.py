"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated, result_global, result_local, self.label_3_0, self.label_3_1, self.label_3_2, self.label_3_3, self.label_3_4, \
        self.label_3_5, self.label_3_6, self.label_3_7, self.label_3_8, self.label_3_9, self.label_3_10, self.label_3_11, self.label_3_12, \
        self.label_3_13, self.label_3_14, self.label_3_15, self.label_3_16, self.label_3_17, self.label_3_18, self.label_3_19, self.label_3_20, \
        self.label_3_21, self.label_3_22, self.label_3_23, self.label_3_24, self.label_3_25, self.label_3_26, self.label_3_27, self.label_3_28, \
        self.label_3_29, self.label_3_30, self.label_3_31, self.label_3_32, self.label_3_33, self.label_3_34, self.label_3_35, self.label_3_36, \
        self.label_3_37,self.label_3_38, self.label_3_39, self.label_3_40,self.label_3_41, self.label_3_42, self.label_3_43,self.label_3_44, \
        self.label_3_45, self.label_3_46, self.label_3_47, self.label_3_48, self.label_3_49,self.label_3_50, self.label_3_51, \
        self.result_0, self.result_1, self.result_2, self.result_3, self.result_4,self.result_5 ,self.result_6 ,self.result_7 , self.result_8 , \
        self.result_9 , self.result_10 , self.result_11 ,self.result_12 , self.result_13 , self.result_14 , self.result_15 , self.result_16 , \
        self.result_17 , self.result_18 ,self.result_19 , self.result_20,self.result_21 , self.result_22 , self.result_23 , self.result_24 , \
        self.result_25 , self.result_26 , self.result_27 , self.result_28 , self.result_29 , self.result_30, self.result_31 , self.result_32 , \
        self.result_33 , self.result_34, self.result_35 , self.result_36,self.result_37 , self.result_38,self.result_39 , self.result_40,\
        self.result_41 , self.result_42,self.result_43 , self.result_44,self.result_45 , self.result_46,self.result_47 , self.result_48,\
        self.result_49 , self.result_50,self.result_51 , self.feature_score, self.target, self.index, self.attention_global, self.attention_local = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated
        self.result_global = result_global
        self.result_local = result_local

        # for i in range(self.opt.label_nc):
        #     globals()['self.label_3_' + str(i)] = eval('label_3_%d'% (i))

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def get_local_generated(self):
        return self.result_local

    def get_global_generated(self):
        return self.result_global

    def get_local_attention(self):
        return self.attention_local

    def get_global_attention(self):
        return self.attention_global

    def get_label_3_0_generated(self):
        return self.label_3_0
    def get_label_3_1_generated(self):
        return self.label_3_1
    def get_label_3_2_generated(self):
        return self.label_3_2
    def get_label_3_3_generated(self):
        return self.label_3_3
    def get_label_3_4_generated(self):
        return self.label_3_4
    def get_label_3_5_generated(self):
        return self.label_3_5
    def get_label_3_6_generated(self):
        return self.label_3_6
    def get_label_3_7_generated(self):
        return self.label_3_7
    def get_label_3_8_generated(self):
        return self.label_3_8
    def get_label_3_9_generated(self):
        return self.label_3_9
    def get_label_3_10_generated(self):
        return self.label_3_10
    def get_label_3_11_generated(self):
        return self.label_3_11
    def get_label_3_12_generated(self):
        return self.label_3_12
    def get_label_3_13_generated(self):
        return self.label_3_13
    def get_label_3_14_generated(self):
        return self.label_3_14
    def get_label_3_15_generated(self):
        return self.label_3_15
    def get_label_3_16_generated(self):
        return self.label_3_16
    def get_label_3_17_generated(self):
        return self.label_3_17
    def get_label_3_18_generated(self):
        return self.label_3_18
    def get_label_3_19_generated(self):
        return self.label_3_19
    def get_label_3_20_generated(self):
        return self.label_3_20
    def get_label_3_21_generated(self):
        return self.label_3_21
    def get_label_3_22_generated(self):
        return self.label_3_22
    def get_label_3_23_generated(self):
        return self.label_3_23
    def get_label_3_24_generated(self):
        return self.label_3_24
    def get_label_3_25_generated(self):
        return self.label_3_25
    def get_label_3_26_generated(self):
        return self.label_3_26
    def get_label_3_27_generated(self):
        return self.label_3_27
    def get_label_3_28_generated(self):
        return self.label_3_28
    def get_label_3_29_generated(self):
        return self.label_3_29
    def get_label_3_30_generated(self):
        return self.label_3_30
    def get_label_3_31_generated(self):
        return self.label_3_31
    def get_label_3_32_generated(self):
        return self.label_3_32
    def get_label_3_33_generated(self):
        return self.label_3_33
    def get_label_3_34_generated(self):
        return self.label_3_34
    def get_label_3_7_generated(self):
        return self.result_7
    def get_label_3_8_generated(self):
        return self.result_8
    def get_label_3_11_generated(self):
        return self.result_11
    def get_label_3_21_generated(self):
        return self.result_21
    def get_label_3_27_generated(self):
        return self.result_27
    def get_label_3_33_generated(self):
        return self.result_33
    def get_label_3_34_generated(self):
        return self.result_34
    def get_label_3_35_generated(self):
        return self.result_35
    def get_label_3_36_generated(self):
        return self.result_36
    def get_label_3_37_generated(self):
        return self.result_37
    def get_label_3_38_generated(self):
        return self.result_38
    def get_image_3_39_generated(self):
        return self.result_39
    def get_label_3_40_generated(self):
        return self.result_40
    def get_label_3_41_generated(self):
        return self.result_41
    def get_label_3_42_generated(self):
        return self.result_42
    def get_label_3_43_generated(self):
        return self.result_43
    def get_label_3_44_generated(self):
        return self.result_44
    def get_label_3_45_generated(self):
        return self.result_45
    def get_label_3_46_generated(self):
        return self.result_46
    def get_label_3_47_generated(self):
        return self.result_47
    def get_label_3_48_generated(self):
        return self.result_48
    def get_label_3_49_generated(self):
        return self.result_49
    def get_label_3_50_generated(self):
        return self.result_50
    def get_label_3_51_generated(self):
        return self.result_51

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
