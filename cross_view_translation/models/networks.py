import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy, collections
from numpy import inf
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'global':
        netG = ResnetGenerator_global(input_nc, output_nc, ngf, n_blocks=9)
    elif which_model_netG == 'fusion':
        netG = ResnetGenerator_fusion(input_nc, output_nc, ngf, n_blocks=9)
    elif which_model_netG == 'local':
        netG = ResnetGenerator_local(input_nc, output_nc, ngf, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    # print('Total number of parameters: %d' % num_params)
    print('Total number of parameters : %.5f M' % (num_params / 1e6))

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        print_network(self.vgg)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def custom_replace(tensor, label):
    # we create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone()
    # res = tensor
    res[tensor==label] = 1
    res[tensor!=label] = 0
    return res

class ResnetGenerator_local(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator_local, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.conv1 = nn.Conv2d(input_nc, 64, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        # self.conv4_norm = nn.InstanceNorm2d(512)
        # self.conv5 = nn.Conv2d(512, 1024, 3, 2, 1)
        # self.conv5_norm = nn.InstanceNorm2d(1024)

        # self.resnet_blocks = []
        # for i in range(n_blocks):
        #     self.resnet_blocks.append(resnet_block(1024, 3, 1, 1))
        #     self.resnet_blocks[i].weight_init(0, 0.02)

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

        # self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        # self.deconv1 = nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1)
        # self.deconv1_norm = nn.InstanceNorm2d(512)
        # self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        # self.deconv2_norm = nn.InstanceNorm2d(256)
        self.deconv3_local = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_local = nn.InstanceNorm2d(128)
        self.deconv4_local = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_local = nn.InstanceNorm2d(64)
        # self.deconv5 = nn.Conv2d(64, output_nc, 7, 1, 0)

        self.deconv5_1 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_2 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_3 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_4 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_5 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_6 = nn.Conv2d(64, output_nc, 7, 1, 0)
        self.deconv5_7 = nn.Conv2d(64, output_nc, 7, 1, 0)



        self.deconv3_global = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_global = nn.InstanceNorm2d(128)
        self.deconv4_global = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_global = nn.InstanceNorm2d(64)
        self.deconv5_global = nn.Conv2d(64, output_nc, 7, 1, 0)


        self.deconv3_attention = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm_attention = nn.InstanceNorm2d(128)
        self.deconv4_attention = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv4_norm_attention = nn.InstanceNorm2d(64)
        self.deconv5_attention = nn.Conv2d(64, 2, 1, 1, 0)

        # self.deconv6 = nn.Conv2d(60, output_nc, 1, 1, 0)
        # self.conv1_2 = nn.Conv2d(1, 1, 1, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, mask):
            mask_1 = custom_replace(mask[:, 0:1, :, :], 10)
            mask_2 = custom_replace(mask[:, 0:1, :, :], 20)
            mask_3 = custom_replace(mask[:, 0:1, :, :], 30)
            mask_4 = custom_replace(mask[:, 0:1, :, :], 40)
            mask_5 = custom_replace(mask[:, 0:1, :, :], 50)
            mask_6 = custom_replace(mask[:, 0:1, :, :], 60)

            # image_class_1_number=torch.sum(mask_1).item()
            # image_class_2_number=torch.sum(mask_2).item()
            # image_class_3_number=torch.sum(mask_3).item()
            # image_class_4_number=torch.sum(mask_4).item()
            # image_class_5_number=torch.sum(mask_5).item()
            # image_class_6_number=torch.sum(mask_6).item()
            # # print(class_1_number)
            #
            # # print(class_1_number)
            # # class_number_total= class_1_number+ class_2_number + class_3_number + class_4_number + class_5_number + class_6_number + class_7_number
            # # print(class_number_total)
            # image_class_number = [image_class_1_number, image_class_2_number, image_class_3_number, image_class_4_number, image_class_5_number, image_class_6_number]
            # # print(class_number)
            # image_class_number = numpy.true_divide(1, image_class_number)
            # image_class_number[image_class_number == inf] = 0
            #
            # image_class_number[image_class_number > 0] = 1
            # print(image_class_number)

            # print(mask_20.size())
            mask_1_64 = mask_1.repeat(1, 64, 1, 1)
            # print(mask_1_64.size())
            mask_2_64 = mask_2.repeat(1, 64, 1, 1)
            mask_3_64 = mask_3.repeat(1, 64, 1, 1)
            mask_4_64 = mask_4.repeat(1, 64, 1, 1)
            mask_5_64 = mask_5.repeat(1, 64, 1, 1)
            mask_6_64 = mask_6.repeat(1, 64, 1, 1)

            mask_1_3 = mask_1.repeat(1, 3, 1, 1)
            mask_2_3 = mask_2.repeat(1, 3, 1, 1)
            mask_3_3 = mask_3.repeat(1, 3, 1, 1)
            mask_4_3 = mask_4.repeat(1, 3, 1, 1)
            mask_5_3 = mask_5.repeat(1, 3, 1, 1)
            mask_6_3 = mask_6.repeat(1, 3, 1, 1)

            x = F.pad(input, (3, 3, 3, 3), 'reflect')
            x = F.relu(self.conv1_norm(self.conv1(x)))
            x = F.relu(self.conv2_norm(self.conv2(x)))
            x = F.relu(self.conv3_norm(self.conv3(x)))
            # print(x.size()) [4 256 64 64]
            # x = F.relu(self.conv4_norm(self.conv4(x)))
            # x = F.relu(self.conv5_norm(self.conv5(x)))

            x = self.resnet_blocks1(x)
            x = self.resnet_blocks2(x)
            x = self.resnet_blocks3(x)
            x = self.resnet_blocks4(x)
            x = self.resnet_blocks5(x)
            x = self.resnet_blocks6(x)
            x = self.resnet_blocks7(x)
            x = self.resnet_blocks8(x)
            middle_x = self.resnet_blocks9(x)

            print(middle_x.size())
            # x = F.relu(self.deconv1_norm(self.deconv1(x)))
            # x = F.relu(self.deconv2_norm(self.deconv2(x)))
            x_local = F.relu(self.deconv3_norm_local(self.deconv3_local(middle_x)))
            print(x_local.size())
            x_feature_local = F.relu(self.deconv4_norm_local(self.deconv4_local(x_local)))
            print(x_feature_local.size())
            # attention = x[:, :1, :, :]
            # print(attention.size())
            # attention = self.conv1_2(attention)

            # print(x_feature.size())
            # [4,64,256,256]

            label_1 = x_feature_local*mask_1_64
            print(mask_1_64.size())
            # print(x.size())
            # print(mask_1_64.size())
            label_2 = x_feature_local*mask_2_64
            label_3 = x_feature_local*mask_3_64
            label_4 = x_feature_local*mask_4_64
            label_5 = x_feature_local*mask_5_64
            label_6 = x_feature_local*mask_6_64


            label_1 = F.pad(label_1, (3, 3, 3, 3), 'reflect')
            label_2 = F.pad(label_2, (3, 3, 3, 3), 'reflect')
            label_3 = F.pad(label_3, (3, 3, 3, 3), 'reflect')
            label_4 = F.pad(label_4, (3, 3, 3, 3), 'reflect')
            label_5 = F.pad(label_5, (3, 3, 3, 3), 'reflect')
            label_6 = F.pad(label_6, (3, 3, 3, 3), 'reflect')


            result_1 = torch.tanh(self.deconv5_1(label_1))
            print(result_1.size())
            result_2 = torch.tanh(self.deconv5_2(label_2))
            result_3 = torch.tanh(self.deconv5_3(label_3))
            result_4 = torch.tanh(self.deconv5_4(label_4))
            result_5 = torch.tanh(self.deconv5_5(label_5))
            result_6 = torch.tanh(self.deconv5_6(label_6))

            result_local = result_1 + result_2 + \
                           result_3 + result_4 + \
                           result_5 + result_6
            # print(result_local.size())

            x_global = F.relu(self.deconv3_norm_global(self.deconv3_global(middle_x)))
            x_global = F.relu(self.deconv4_norm_global(self.deconv4_global(x_global)))
            x_global = F.pad(x_global, (3, 3, 3, 3), 'reflect')
            result_global = torch.tanh(self.deconv5_global(x_global))
            # print(middle_x.size())
            x_attention = F.relu(self.deconv3_norm_attention(self.deconv3_attention(middle_x)))
            # print(x_attention.size())
            x_attention = F.relu(self.deconv4_norm_attention(self.deconv4_attention(x_attention)))
            result_attention = self.deconv5_attention(x_attention)
            # print(result_attention.size())
            softmax_ = torch.nn.Softmax(dim=1)
            result_attention = softmax_(result_attention)

            attention_local = result_attention[:, 0:1, :, :]
            attention_global = result_attention[:, 1:2, :, :]

            attention_local = attention_local.repeat(1, 3, 1, 1)
            attention_global = attention_global.repeat(1, 3, 1, 1)

            final_result =  attention_local* result_local + attention_global * result_global

            # print("I am here.")

            return mask_1_3, mask_2_3, mask_3_3, mask_4_3, mask_5_3, mask_6_3, result_1,result_2, result_3, \
                   result_4, result_5, result_6, result_local, attention_local, result_global, attention_global, final_result


class ResnetGenerator_global(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator_global, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.conv1 = nn.Conv2d(input_nc, 64, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        # self.conv4_norm = nn.InstanceNorm2d(512)
        # self.conv5 = nn.Conv2d(512, 1024, 3, 2, 1)
        # self.conv5_norm = nn.InstanceNorm2d(1024)

        # self.resnet_blocks = []
        # for i in range(n_blocks):
        #     self.resnet_blocks.append(resnet_block(1024, 3, 1, 1))
        #     self.resnet_blocks[i].weight_init(0, 0.02)

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

        # self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        # self.deconv1 = nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1)
        # self.deconv1_norm = nn.InstanceNorm2d(512)
        # self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        # self.deconv2_norm = nn.InstanceNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv3_norm = nn.InstanceNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64+1, 3, 2, 1, 1)
        self.deconv4_norm = nn.InstanceNorm2d(64)
        self.deconv5 = nn.Conv2d(64, output_nc, 7, 1, 0)

        # self.deconv5_1 = nn.Conv2d(64, output_nc, 7, 1, 0)
        # self.deconv5_2 = nn.Conv2d(64, output_nc, 7, 1, 0)
        # self.deconv5_3 = nn.Conv2d(64, output_nc, 7, 1, 0)
        # self.deconv5_4 = nn.Conv2d(64, output_nc, 7, 1, 0)
        # self.deconv5_5 = nn.Conv2d(64, output_nc, 7, 1, 0)
        # self.deconv5_6 = nn.Conv2d(64, output_nc, 7, 1, 0)
        # self.deconv5_7 = nn.Conv2d(64, output_nc, 7, 1, 0)

        # self.deconv6 = nn.Conv2d(60, output_nc, 1, 1, 0)

        # self.conv1_1 = nn.Conv2d(1, 1, 1, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):

        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        # print(x.size()) [4 256 64 64]
        # x = F.relu(self.conv4_norm(self.conv4(x)))
        # x = F.relu(self.conv5_norm(self.conv5(x)))

        x = self.resnet_blocks1(x)
        x = self.resnet_blocks2(x)
        x = self.resnet_blocks3(x)
        x = self.resnet_blocks4(x)
        x = self.resnet_blocks5(x)
        x = self.resnet_blocks6(x)
        x = self.resnet_blocks7(x)
        x = self.resnet_blocks8(x)
        x = self.resnet_blocks9(x)

        # x = F.relu(self.deconv1_norm(self.deconv1(x)))
        # x = F.relu(self.deconv2_norm(self.deconv2(x)))
        x = F.relu(self.deconv3_norm(self.deconv3(x)))
        x = F.relu(self.deconv4_norm(self.deconv4(x)))
        attention = x[:, :1, :, :]
        # attention = self.conv1_1(attention)
        x_feature = x[:, 1:, :, :]
        # print(x_feature.size())
        # [4,64,256,256]

        x_feature = F.pad(x_feature, (3, 3, 3, 3), 'reflect')

        result_global = torch.tanh(self.deconv5(x_feature))
        # print(result_1.size())



        # result_local=torch.cat((result_1, result_2, result_3, result_4, result_5, result_6, result_7,result_8,result_9,result_10,\
        #                result_11,result_12,result_13,result_14,result_15,result_16,result_17,result_18,result_19,result_20), 1)
        # # print(result_local.size())
        # result_local=torch.tanh(self.deconv6(result_local))


        # sigmoid_ = torch.nn.Sigmoid()
        # attention = sigmoid_(attention)
        # attention = attention.repeat(1, 3, 1, 1)

        return result_global, attention

class ResnetGenerator_fusion(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator_fusion, self).__init__()

        # self.conv1_1 = nn.Conv2d(input_nc, output_nc, 1, 1, 0)
        # self.conv1_2 = nn.Conv2d(input_nc, output_nc, 1, 1, 0)

    # forward method
    def forward(self, gloabl_result, local_result, attention_global, attention_local):
        attention=torch.cat((attention_global, attention_local), 1)
        softmax_ = torch.nn.Softmax(dim=1)
        attention_2 = softmax_(attention)
        attention_local = attention_2[:, 0:1, :, :]
        attention_global = attention_2[:, 1:2, :, :]
        result =  attention_local* local_result +  attention_global *  gloabl_result

        local_attention = attention_local.repeat(1, 3, 1, 1)
        global_attention = attention_global.repeat(1, 3, 1, 1)

        return result, local_attention, global_attention

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

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
