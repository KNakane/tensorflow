import os, sys
import argparse
import tensorflow as tf
from CNN.lenet import LeNet, VGG
from CNN.resnet import ResNet18, ResNet34
from AutoEncoder.model import AutoEncoder, VAE
from GAN.gan import GAN
from GAN.dcgan import DCGAN
from dataset.load import Load
from trainer.trainer import Trainer, AE_Trainer, GAN_Trainer
from collections import OrderedDict


def image_recognition(args):
    message = OrderedDict({
        "Network": args.network,
        "data": args.data,
        "epoch":args.n_epoch,
        "batch_size": args.batch_size,
        "Optimizer":args.opt,
        "learning_rate":args.lr,
        "l2_norm": args.l2_norm,
        "Augmentation": args.aug})

    data = Load(args.data)
    model = eval(args.network)(name=args.network, out_dim=data.output_dim, lr=args.lr, opt=args.opt, l2_reg=args.l2_norm)

    #training
    trainer = Trainer(args, message, data, model, args.network)
    trainer.train()
    return

def construction_image(args):
    message = OrderedDict({
        "Network": args.network,
        "data": args.data,
        "epoch":args.n_epoch,
        "batch_size": args.batch_size,
        "Denoising":args.denoise,
        "Optimizer":args.opt,
        "learning_rate":args.lr,
        "l2_norm": args.l2_norm,
        "Augmentation": args.aug})

    data = Load(args.data)
    model = eval(args.network)(name=args.network, size=data.size, channel=data.channel, out_dim=data.output_dim*2, lr=args.lr, opt=args.opt, l2_reg=args.l2_norm)

    #training
    trainer = AE_Trainer(args, message, data, model, args.network)
    trainer.train()
    return

def GAN_fn(args):
    message = OrderedDict({
        "Network": args.network,
        "Conditional": args.conditional,
        "data": args.data,
        "z_dim": args.z_dim,
        "epoch":args.n_epoch,
        "batch_size": args.batch_size,
        "Optimizer":args.opt,
        "learning_rate":args.lr,
        "n_disc_update":args.n_disc_update,
        "l2_norm":args.l2_norm,
        "Augmentation": args.aug})

    data = Load(args.data)
    model = eval(args.network)(z_dim=args.z_dim,
                                size=data.size,
                                channel=data.channel,
                                name=args.network,
                                class_num=data.output_dim,
                                lr=args.lr, opt=args.opt, l2_reg=args.l2_norm)

    #training
    trainer = GAN_Trainer(args, message, data, model, args.network)
    trainer.train()
    return


def main(args):
    if args.network == 'LeNet' or args.network == 'VGG' or args.network == 'ResNet18' or args.network == 'ResNet34':
        image_recognition(args)
    elif args.network == 'AutoEncoder' or args.network == 'VAE':
        construction_image(args)
    elif args.network == 'GAN' or args.network == 'DCGAN':
        GAN_fn(args)
    else:
        raise NotImplementedError()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='LeNet', type=str, choices=['LeNet','VGG','ResNet18','ResNet34','AutoEncoder','VAE','GAN','DCGAN'])
    parser.add_argument('--data', default='mnist', type=str, choices=['mnist','cifar10','cifar100','kuzushiji'])
    parser.add_argument('--n_epoch', default=1000, type=int, help='Input max epoch')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Input learning rate')
    parser.add_argument('--opt', default='SGD', type=str, choices=['SGD','Momentum','Adadelta','Adagrad','Adam','RMSProp'])
    parser.add_argument('--aug', default=None, type=str, choices=['shift','mirror','rotate','shift_rotate','cutout','random_erace'])
    parser.add_argument('--denoise', action='store_true', help='True : Denoising AE, False : standard AE')
    parser.add_argument('--l2_norm', action='store_true', help='L2 normalization or not')
    parser.add_argument('--z_dim', default=100, type=int, help='Latent z dimension')
    parser.add_argument('--conditional', action='store_true', help='Conditional true or false')
    parser.add_argument('--n_disc_update', default=1, type=int, help='Learning times for discriminator')
    parser.add_argument('--init_model', default=None, type=str, help='Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    parser.add_argument('--checkpoints_to_keep', default=5, type=int, help='checkpoint keep count')
    parser.add_argument('--keep_checkpoint_every_n_hours', default=1, type=int, help='checkpoint create hour')
    parser.add_argument('--save_checkpoint_steps', default=100, type=int, help='save checkpoint step')
    args = parser.parse_args()
    main(args)