import os
import argparse
import logging
import importlib
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import _init_paths
from utils.kd_model import KD_model
from libs.datasets import AcdcDataset
from libs.network import U_Net_teacher
from libs.datasets import joint_augment as joint_augment
from libs.datasets import augment as standard_augment
import train_utils.train_utils as train_utils
from train_utils.train_utils import load_checkpoint
from utils.init_net import init_weights
from utils.comm import get_rank, synchronize
import numpy as np
import random
from libs.configs.acdc.config import cfg


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--local_rank", type=int, default=0, required=True, help="device_ids of DistributedDataParallel")
parser.add_argument("--batch_size", type=int, default=20, required=False, help="batch size for training")
parser.add_argument("--epochs", type=int, default=200, required=False, help="Number of epochs to train for")
parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--ckpt_save_interval", type=int, default=5, help="number of training epochs")
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument('--mgpus', type=str, default=None, help='whether to use multiple gpu')
parser.add_argument("--ckpt", type=str, default=None, help="continue training from this checkpoint")
parser.add_argument("--ckpt_t", type=str, default=None, help="teacher's checkpoint for KD")
parser.add_argument("--imgmean", type=bool, default=False, help="continue training from this checkpoint")
parser.add_argument("--kd", type=bool, default=False, help="")
parser.add_argument('--train_with_eval', action='store_true', default=True, help='whether to train with evaluation')
args = parser.parse_args()

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

if args.mgpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.mgpus


def create_logger(log_file, dist_rank):
    if dist_rank > 0:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)
        return logger
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def create_dataloader(logger):
    transform = standard_augment.Compose([
        standard_augment.to_Tensor()])
    train_joint_transform = joint_augment.Compose([
        joint_augment.To_PIL_Image(),
        joint_augment.RandomAffine(0, translate=(0.125, 0.125)),
        joint_augment.RandomRotate((-180, 180)),
        joint_augment.FixResize(256)
    ])
    target_transform = standard_augment.Compose([
        standard_augment.to_Tensor()])
    train_set = AcdcDataset(data_path=cfg.DATASET.TRAIN_LIST,
                              joint_augment=train_joint_transform,
                              augment=transform, target_augment=target_transform, split='train')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                    num_replicas=dist.get_world_size(),
                                                                    rank=dist.get_rank())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.workers, shuffle=False, sampler=train_sampler)

    if args.train_with_eval:
        eval_transform = joint_augment.Compose([
            joint_augment.To_PIL_Image(),
            joint_augment.FixResize(256),
            joint_augment.To_Tensor()])
        test_set = AcdcDataset(data_path=cfg.DATASET.TEST_LIST,
                                 joint_augment=eval_transform, split='test')

        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set,
                                                                       num_replicas=dist.get_world_size(),
                                                                       rank=dist.get_rank())
        test_loader = DataLoader(test_set, batch_size=args.batch_size, pin_memory=True,
                                 num_workers=args.workers, shuffle=False, sampler=test_sampler)
    else:
        test_loader = None

    return train_loader, test_loader


def create_optimizer(model):
    if cfg.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                              momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise NotImplementedError
    return optimizer

def create_scheduler(model, optimizer, total_steps, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.LR_DECAY
        return max(cur_decay, cfg.TRAIN.LR_CLIP / cfg.TRAIN.LR)

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    return lr_scheduler

def create_model(cfg):
    network = cfg.TRAIN.NET

    module = 'libs.network.' + network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    mod_func = importlib.import_module('libs.network.train_functions')
    net_func = getattr(mod, model)
    net = net_func(img_ch=cfg.DATASET.IMG_CH, num_class=cfg.DATASET.NUM_CLASS)
    train_func = getattr(mod_func, 'model_fn_decorator')

    return net, train_func


def train():
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    synchronize()

    # create dataloader & network & optimizer
    model, model_fn_decorator = create_model(cfg)
    init_weights(model, init_type='kaiming')
    model.cuda()
    if args.kd:
        teacher_model = U_Net_teacher(cfg.DATASET.IMG_CH, cfg.DATASET.NUM_CLASS)
        model_file = args.ckpt_t
        teacher_model = teacher_model.cuda()
        model_data = torch.load(model_file)
        teacher_model.load_state_dict(model_data['model_state'])
        teacher_model.eval()
        model = KD_model(model, teacher_model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    root_result_dir = args.output_dir
    os.makedirs(root_result_dir, exist_ok=True)
    print("**********************root_result_dir:", root_result_dir)

    log_file = os.path.join(root_result_dir, "log_train.txt")
    logger = create_logger(log_file, get_rank())
    logger.info("**********************Start logging**********************")

    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info("CUDA_VISIBLE_DEVICES=%s" % gpu_list)

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))

    logger.info("***********************config infos**********************")
    for key, val in vars(cfg).items():
        logger.info("{:16} {}".format(key, val))

    # log tensorboard
    if get_rank() == 0:
        tb_log = SummaryWriter(log_dir=os.path.join(root_result_dir, "tensorboard"))
    else:
        tb_log = None

    train_loader, test_loader = create_dataloader(logger)

    optimizer = create_optimizer(model)
    # load checkpoint if it is possible
    start_epoch = it = best_res = 0
    last_epoch = -1
    if args.ckpt is not None:
        pure_model = model.module if isinstance(model, (
            torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
        it, start_epoch, best_res = load_checkpoint(pure_model, optimizer, args.ckpt, logger)
        last_epoch = start_epoch + 1

    lr_scheduler = create_scheduler(model, optimizer, total_steps=len(train_loader) * args.epochs,
                                    last_epoch=last_epoch)

    criterion = nn.CrossEntropyLoss()

    # start training
    logger.info('**********************Start training**********************')
    ckpt_dir = os.path.join(root_result_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    print('**********************ckpt_dir:', ckpt_dir)
    trainer = train_utils.Trainer(model,
                                  model_fn=model_fn_decorator(),
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  ckpt_dir=ckpt_dir,
                                  lr_scheduler=lr_scheduler,
                                  model_fn_eval=model_fn_decorator(),
                                  tb_log=tb_log,
                                  logger=logger,
                                  eval_frequency=1,
                                  grad_norm_clip=cfg.TRAIN.GRAD_NORM_CLIP,
                                  cfg=cfg, imgmean=args.imgmean, kd=args.kd)

    trainer.train(start_it=it,
                  start_epoch=start_epoch,
                  n_epochs=args.epochs,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  ckpt_save_interval=args.ckpt_save_interval,
                  lr_scheduler_each_iter=False,
                  best_res=best_res)

    logger.info('**********************End training**********************')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# python -m torch.distributed.launch --nproc_per_node 1 --master_port $RANDOM tools/train.py --batch_size 16 --mgpus 0 --output_dir logs/... --imgmean
if __name__ == "__main__":
    setup_seed(20)

    start = time.process_time()
    train()
    end = time.process_time()
    print("The function run time is : %.03f seconds" % (end - start))
    save_path = os.path.join(args.output_dir, 'time_rec.txt')
    file = open(save_path, mode='a')
    file.write(str(end - start) + '\n')
    file.close()
