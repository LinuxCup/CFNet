import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb
from torch.nn.utils import clip_grad_norm_

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets

from models import *

import tqdm
import logging
import importlib
from utils.logger import config_logger
from utils import builder
from tensorboardX import SummaryWriter


import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = torch.distributed.get_world_size()
    # pdb.set_trace()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


def train_fp16(epoch, end_epoch, args, model, train_loader, optimizer, scheduler, logger, log_frequency, tb_log):
    scaler = torch.cuda.amp.GradScaler()
    rank = torch.distributed.get_rank()
    model.train()
    loss_tb = AverageMeter()
    lr_tb = AverageMeter()
    # torch.autograd.set_detect_anomaly(True)
    for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset,\
        pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw, seq_id, fn) in tqdm.tqdm(enumerate(train_loader)):
        #pdb.set_trace()
        with torch.cuda.amp.autocast():
            loss = model(pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset,\
                pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw)
        
        # sync all gpus
        reduced_loss = reduce_tensor(loss)
        # pdb.set_trace()

        # with torch.autograd.detect_anomaly():
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if (i % log_frequency == 0) and rank == 0:
            string = 'Epoch: [{}]/[{}]; Iteration: [{}]/[{}]; lr: {}'.format(epoch, end_epoch,\
                i, len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'])
            
            string = string + '; loss: {}'.format(reduced_loss.item() / torch.distributed.get_world_size())
            logger.info(string)
        loss_tb.update(reduced_loss)
        lr_tb.update(optimizer.state_dict()['param_groups'][0]['lr'])
    if rank == 0:
        tb_log.add_scalar('train/loss', loss_tb.avg, epoch)
        tb_log.add_scalar('train/lr', lr_tb.avg, epoch)
        


def train(epoch, end_epoch, args, model, train_loader, optimizer, scheduler, logger, log_frequency):
    rank = torch.distributed.get_rank()
    model.train()
    for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset,\
        pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw, seq_id, fn) in tqdm.tqdm(enumerate(train_loader)):
        #pdb.set_trace()
        loss = model(pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset,\
            pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw)
        
        # sync all gpus
        reduced_loss = reduce_tensor(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (i % log_frequency == 0) and rank == 0:
            string = 'Epoch: [{}]/[{}]; Iteration: [{}]/[{}]; lr: {}'.format(epoch, end_epoch,\
                i, len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'])
            
            string = string + '; loss: {}'.format(reduced_loss.item() / torch.distributed.get_world_size())
            logger.info(string)


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()

    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    os.system('mkdir -p {}'.format(model_prefix))

    # start logging
    config_logger(os.path.join(save_path, "log.txt"))
    logger = logging.getLogger()

    # reset dist
    device = torch.device('cuda:{}'.format(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # reset random seed
    seed = rank * pDataset.Train.num_workers + 50051
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # pdb.set_trace()
    
    tb_log = SummaryWriter(log_dir=str(os.path.join(save_path, "tensorboard"))) if args.local_rank == 0 else None

    # define dataloader
    train_dataset = eval('datasets.{}.DataloadTrain'.format(pDataset.Train.data_src))(pDataset.Train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                            batch_size=pGen.batch_size_per_gpu,
                            shuffle=(train_sampler is None),
                            num_workers=pDataset.Train.num_workers,
                            sampler=train_sampler,
                            pin_memory=True)

    print("rank: {}/{}; batch_size: {}".format(rank, world_size, pGen.batch_size_per_gpu))

    # define model
    base_net = eval(pModel.prefix).AttNet(pModel)
    # load pretrain model
    pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(pModel.pretrain.pretrain_epoch))
    if os.path.exists(pretrain_model):
        base_net.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
        logger.info("Load model from {}".format(pretrain_model))
        # pdb.set_trace()
        # it, start_epoch = base_net.load_params_with_optimizer(pretrain_model, to_cpu='cpu', optimizer=optimizer, logger=logger)

    base_net = nn.SyncBatchNorm.convert_sync_batchnorm(base_net)
    model = torch.nn.parallel.DistributedDataParallel(base_net.to(device),
                                                    device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True)

    # define optimizer
    optimizer = builder.get_optimizer(pOpt, model)

    # define scheduler
    per_epoch_num_iters = len(train_loader)
    scheduler = builder.get_scheduler(optimizer, pOpt, per_epoch_num_iters)

    if rank == 0:
        logger.info(model)
        logger.info(optimizer)
        logger.info(scheduler)

    # start training
    for epoch in range(pOpt.schedule.begin_epoch, pOpt.schedule.end_epoch):
        train_sampler.set_epoch(epoch)
        if pGen.fp16:
            train_fp16(epoch, pOpt.schedule.end_epoch, args, model, train_loader, optimizer, scheduler, logger, pGen.log_frequency, tb_log)
        else:
            train(epoch, pOpt.schedule.end_epoch, args, model, train_loader, optimizer, scheduler, logger, pGen.log_frequency)

        # save model
        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(model_prefix, '{}-model.pth'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)