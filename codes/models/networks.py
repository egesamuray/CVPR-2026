# codes/models/networks.py
import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

logger = logging.getLogger('base')

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None: m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None: m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None: m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None: m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None: m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None: m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        net.apply(functools.partial(weights_init_normal, std=std))
    elif init_type == 'kaiming':
        net.apply(functools.partial(weights_init_kaiming, scale=scale))
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))

def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'sr_resnet':
        import models.modules.architecture as arch
        netG = arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                             nb=opt_net['nb'], upscale=opt['scale'], norm_type=opt_net['norm_type'],
                             act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'sft_arch':
        import models.modules.sft_arch as sft_arch
        netG = sft_arch.SFT_Net()

    elif which_model == 'RRDB_net':
        import models.modules.architecture as arch
        netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt['scale'],
                            norm_type=opt_net['norm_type'], act_type='leakyrelu',
                            mode=opt_net['mode'], upsample_mode='upconv')

    elif which_model == 'SR3UNet':
        import models.modules.diffusion_net as diff_net
        netG = diff_net.SR3UNet(
            in_ch=opt_net['in_nc'],
            out_ch=opt_net['out_nc'],
            base_nf=opt_net.get('nf', 64),
            num_res_blocks=opt_net.get('num_res_blocks', 2),
            num_classes=opt_net.get('num_classes', None),
            dropout=opt_net.get('dropout', 0.0)    # <── NEW
        )
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['is_train']:
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
        print(netG)
    return netG
