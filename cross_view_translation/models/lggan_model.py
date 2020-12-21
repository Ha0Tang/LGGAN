import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import itertools
from PIL import Image
import numpy as np
from matplotlib import cm

class LGGANModel(BaseModel):
    def name(self):
        return 'LGGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='instance')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='resnet_9blocks')
        parser.add_argument('--REGULARIZATION', type=float, default=1e-6)
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--cyc_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for vgg loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN_D1','G_GAN_D2', 'G_L1', 'G_VGG', 'G_TV', 'G' , 'D1_real', 'D1_fake','D1', 'D2_real', 'D2_fake','D2']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'real_D', 'm1', 'g1', 'real_g1', 'm2', 'g2', 'real_g2',
                             'm3', 'g3', 'real_g3','m4', 'g4', 'real_g4', 'm5', 'g5', 'real_g5', 'm6', 'g6', 'real_g6',
                             'fake_B_local', 'attention_global', 'fake_B_global', 'attention_local', 'fake_B', 'real_B']
        # self.visual_names = ['fake_B', 'fake_D']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G','D1','D2']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(6, 3, opt.ngf,'local', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netGl = networks.define_G(6, 3, opt.ngf,'local', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netGc = networks.define_G(6, 3, 4,opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netGf = networks.define_G(3, 3, opt.ngf,'fusion', opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD1 = networks.define_D(6, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks.define_D(6, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # initialize optimizers
            self.optimizers = []
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD1.parameters(), self.netD2.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.real_D = input['D'].to(self.device)
        self.real_E = input['E'].to(self.device)
        self.real_F = input['F'].to(self.device)
        self.real_F_mask = (self.real_F*127.5)+127.5
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        combine_AD=torch.cat((self.real_A, self.real_D), 1)
        # combine_ACD=torch.cat((self.real_A, self.real_D), 1)
        self.m1, self.m2, self.m3, self.m4, self.m5, self.m6, self.g1, \
        self.g2, self.g3, self.g4, self.g5, self.g6, \
        self.fake_B_local, self.attention_local, self.fake_B_global, \
        self.attention_global, self.fake_B = self.netG(combine_AD, self.real_F_mask)

        # self.attention_global=torch.squeeze(self.attention_global, dim=0)
        # self.attention_global= self.attention_global.data.cpu().numpy()*255
        # self.attention_local=torch.squeeze(self.attention_local, dim = 0)
        # self.attention_local = self.attention_local.data.cpu().numpy()*255
        # self.attention_global = torch.tanh(self.attention_global)
        #
        # self.attention_local = torch.tanh(self.attention_local)
        # print(self.attention_global.size())
        # print(self.attention_global.type())
        # self.attention_global1 = self.attention_global.copy()
        # self.attention_local1 = self.attention_local.copy()
        # self.attention_global1 = Image.fromarray(np.uint8(cm.jet(torch.squeeze(self.attention_global1,dim=[0,1]).data.cpu().numpy())*255)).convert('RGB')
        # self.attention_local1 = Image.fromarray(np.uint8(cm.jet(torch.squeeze(self.attention_local1,dim=[0,1]).data.cpu().numpy())*255)).convert('RGB')

        self.real_g1=self.real_B * self.m1
        self.real_g2=self.real_B * self.m2
        self.real_g3=self.real_B * self.m3
        self.real_g4=self.real_B * self.m4
        self.real_g5=self.real_B * self.m5
        self.real_g6=self.real_B * self.m6

    def backward_D1(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB_global = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B_global), 1))
        pred_D1_fake_global = self.netD1(fake_AB_global.detach())
        self.loss_D1_fake_global = self.criterionGAN(pred_D1_fake_global, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real_D1 = self.netD1(real_AB)
        self.loss_D1_real = self.criterionGAN(pred_real_D1, True)

        # Combined loss
        self.loss_D1 = (self.loss_D1_fake_global + self.loss_D1_real) * 0.5


        fake_AB_local = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B_local), 1))
        pred_D1_fake_local = self.netD1(fake_AB_local.detach())
        self.loss_D1_fake_local = self.criterionGAN(pred_D1_fake_local, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real_D1 = self.netD1(real_AB)
        self.loss_D1_real = self.criterionGAN(pred_real_D1, True)

        # Combined loss
        self.loss_D1 = (self.loss_D1_fake_local + self.loss_D1_real) * 0.5 + self.loss_D1
        #
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_D1_fake = self.netD1(fake_AB.detach())
        self.loss_D1_fake = self.criterionGAN(pred_D1_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real_D1 = self.netD1(real_AB)
        self.loss_D1_real = self.criterionGAN(pred_real_D1, True)

        # Combined loss
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5 + self.loss_D1

        self.loss_D1.backward()

    def backward_D2(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_DB_global = self.fake_AB_pool.query(torch.cat((self.real_D, self.fake_B_global), 1))
        pred_D2_fake_global = self.netD2(fake_DB_global.detach())
        self.loss_D2_fake_global = self.criterionGAN(pred_D2_fake_global, False)

        # Real
        real_DB = torch.cat((self.real_D, self.real_B), 1)
        pred_real_D2 = self.netD2(real_DB)
        self.loss_D2_real = self.criterionGAN(pred_real_D2, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake_global + self.loss_D2_real) * 0.5
        #
        #
        fake_DB_local = self.fake_AB_pool.query(torch.cat((self.real_D, self.fake_B_local), 1))
        pred_D2_fake_local = self.netD2(fake_DB_local.detach())
        self.loss_D2_fake_local = self.criterionGAN(pred_D2_fake_local, False)

        # Real
        real_DB = torch.cat((self.real_D, self.real_B), 1)
        pred_real_D2 = self.netD2(real_DB)
        self.loss_D2_real = self.criterionGAN(pred_real_D2, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake_local + self.loss_D2_real) * 0.5 + self.loss_D2

        fake_DB = self.fake_AB_pool.query(torch.cat((self.real_D, self.fake_B), 1))
        pred_D2_fake = self.netD2(fake_DB.detach())
        self.loss_D2_fake = self.criterionGAN(pred_D2_fake, False)

        # Real
        real_DB = torch.cat((self.real_D, self.real_B), 1)
        pred_real_D2 = self.netD2(real_DB)
        self.loss_D2_real = self.criterionGAN(pred_real_D2, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5 + self.loss_D2


        self.loss_D2.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB_global = torch.cat((self.real_A, self.fake_B_global), 1)
        pred_D1_fake_global = self.netD1(fake_AB_global)
        self.loss_G_GAN_D1 = self.criterionGAN(pred_D1_fake_global, True)

        fake_DB_global = torch.cat((self.real_D, self.fake_B_global), 1)
        pred_D2_fake_global = self.netD2(fake_DB_global)
        self.loss_G_GAN_D2 = self.criterionGAN(pred_D2_fake_global, True)


        fake_AB_local = torch.cat((self.real_A, self.fake_B_local), 1)
        pred_D1_fake_local = self.netD1(fake_AB_local)
        self.loss_G_GAN_D1 = self.criterionGAN(pred_D1_fake_local, True) + self.loss_G_GAN_D1

        fake_DB_local = torch.cat((self.real_D, self.fake_B_local), 1)
        pred_D2_fake_local = self.netD2(fake_DB_local)
        self.loss_G_GAN_D2 = self.criterionGAN(pred_D2_fake_local, True) + self.loss_G_GAN_D2


        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_D1_fake = self.netD1(fake_AB)
        self.loss_G_GAN_D1 = self.criterionGAN(pred_D1_fake, True) + self.loss_G_GAN_D1

        fake_DB = torch.cat((self.real_D, self.fake_B), 1)
        pred_D2_fake = self.netD2(fake_DB)
        self.loss_G_GAN_D2 = self.criterionGAN(pred_D2_fake, True) + self.loss_G_GAN_D2


        self.loss_G_local_L1 = (self.criterionL1(self.g1, self.real_g1) + self.criterionL1(self.g2, self.real_g2) +
                                self.criterionL1(self.g3, self.real_g3) + self.criterionL1(self.g4, self.real_g4) +
                                self.criterionL1(self.g5, self.real_g5) + self.criterionL1(self.g6, self.real_g6)) * self.opt.lambda_L1

        # second, G(A)=B
        self.loss_G_global_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 + \
                                self.criterionL1(self.fake_B_global, self.real_B) * self.opt.lambda_L1 + \
                                self.criterionL1(self.fake_B_local, self.real_B) * self.opt.lambda_L1


        self.loss_G_L1 = self.loss_G_local_L1 + self.loss_G_global_L1

        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_feat + \
                          self.criterionVGG(self.fake_B_global, self.real_B) * self.opt.lambda_feat + \
                          self.criterionVGG(self.fake_B_local, self.real_B) * self.opt.lambda_feat

        self.loss_G_TV =  self.opt.REGULARIZATION * (
                torch.sum(torch.abs(self.fake_B[:, :, :, :-1] - self.fake_B[:, :, :, 1:])) +
                torch.sum(torch.abs(self.fake_B[:, :, :-1, :] - self.fake_B[:, :, 1:, :]))) + \
                          self.opt.REGULARIZATION * (
                torch.sum(torch.abs(self.fake_B_global[:, :, :, :-1] - self.fake_B_global[:, :, :, 1:])) +
                torch.sum(torch.abs(self.fake_B_global[:, :, :-1, :] - self.fake_B_global[:, :, 1:, :]))) + \
                          self.opt.REGULARIZATION * (
                torch.sum(torch.abs(self.fake_B_local[:, :, :, :-1] - self.fake_B_local[:, :, :, 1:])) +
                torch.sum(torch.abs(self.fake_B_local[:, :, :-1, :] - self.fake_B_local[:, :, 1:, :])))

        self.loss_G = self.loss_G_GAN_D1 + self.loss_G_GAN_D2 + self.loss_G_L1 + self.loss_G_VGG + self.loss_G_TV

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad([self.netD1, self.netD2], True)
        self.optimizer_D.zero_grad()
        self.backward_D1()
        self.backward_D2()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD1, self.netD2], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
