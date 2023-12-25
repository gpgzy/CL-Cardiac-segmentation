import numpy as np
import torch
import dataloaders
from utils.losses import abCE_loss, CE_loss, consistency_weight,LovaszLossSoftmax
import models
import json
from trainer_heart import Trainer
from utils import Logger
import torch.multiprocessing as mp
import torch.distributed as dist
# from torchstat import stat
from models.modeling.deeplab import DeepLab as DeepLab_v3p
from models.modeling.unet_model import UNet


#使用resnet18作为backbone
def initkwargs():
    config = json.load(open('configs/heart_cac_deeplabv3+_resnet50_1over8_datalist0.json'))
    config['train_supervised']['batch_size'] = int(config['train_supervised']['batch_size'] / config['n_gpu'])
    config['train_unsupervised']['batch_size'] = int(config['train_unsupervised']['batch_size'] / config['n_gpu'])
    config['val_loader']['batch_size'] = int(config['val_loader']['batch_size'] / config['n_gpu'])
    config['train_supervised']['num_workers'] = int(config['train_supervised']['num_workers'] / config['n_gpu'])
    config['train_unsupervised']['num_workers'] = int(config['train_unsupervised']['num_workers'] / config['n_gpu'])
    config['val_loader']['num_workers'] = int(config['val_loader']['num_workers'] / config['n_gpu'])
    config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
    config['train_supervised']['use_weak_lables'] = config['use_weak_lables']
    config['train_supervised']['data_dir'] = config['data_dir']
    config['train_unsupervised']['data_dir'] = config['data_dir']
    return config


def train(args):
    # print(args)
    config = json.load(open('configs/heart_cac_deeplabv3+_resnet50_1over8_datalist0.json'))
    port = find_free_port()
    config['dist_url'] = f"tcp://127.0.0.1:{port}"
    config['n_node'] = 0  # only support 1 node
    config['world_size'] = config['n_gpu']
    config['rank'] = 0
    dist.init_process_group(backend='nccl', init_method=config['dist_url'], world_size=config['world_size'],
                            rank=config['rank'])
    sup_dataloader = dataloaders.HEART
    unsup_dataloader = dataloaders.PairHEART
    valid_dataloader = dataloaders.HEARTValid
    config_data = initkwargs()
    supervised_loader = sup_dataloader(config_data['train_supervised'])
    unsupervised_loader = unsup_dataloader(config_data['train_unsupervised'])
    val_loader = valid_dataloader(config_data['val_loader'])
    #### Fix iter_per_epoch ####
    print(config_data['train_supervised'])
    iter_per_epoch = 125
    sup_loss = CE_loss
    # sup_loss = LovaszLossSoftmax()
    model = models.CAC(num_classes=4, conf=config['model'],sup_loss=sup_loss, ignore_index=255)
    print(model)
    # stat(DeepLab_v3p(backbone='resnet{}'.format(50)), (3, 512, 512))
    # stat(UNet(3,4), (3, 512, 512))
    # print(sup_loss)
    # 加载预训练好的模型作为初始参数
    # pretrained_dict = torch.load('saved/pretrainedModels/checkpoint.pth')
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print(sup_loss)
    gpu = 0
    train_logger = Logger()
    #resume 设置从断点处开始训练模型
    trainer = Trainer(
        model=model,
        resume=False,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch,
        train_logger=train_logger,
        gpu=gpu,
        test=False
    )
    trainer.train()
if __name__ == '__main__':
    # train()
    mp.spawn(train,nprocs=1)
def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
