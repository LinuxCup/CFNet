import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets

from utils.metric import PanopticEval
from models import *

import pytorch_lib

import collections

import tqdm
import importlib
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

import yaml
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False


save_results = False
def merge_offset_tta(pred_offset):
    '''
    Input:
        pred_offset, (4, N, 3)
    Output:
        pred_offset_result, (N, 3)
    '''
    assert pred_offset.ndim == 3
    assert (pred_offset.shape[0] == 4) or (pred_offset.shape[0] == 1)
    assert pred_offset.shape[2] == 3
    if pred_offset.shape[0] == 4:
        p = 0
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                pred_offset[p, :, 0] *= x_sign
                pred_offset[p, :, 1] *= y_sign
                p += 1
    pred_offset_result = pred_offset.mean(dim=0)
    return pred_offset_result


def val_fp16(epoch, model, val_loader, category_list, save_path, rank=0):
    criterion_pano = PanopticEval(category_list, None, [0], min_points=50)
    model.eval()
    pv_nms = pytorch_lib.PointVoteNMS(model.point_nms_dic)
    f = open(os.path.join(save_path, 'record_fp16_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset, pano_label, seq_id, fn) in tqdm.tqdm(enumerate(val_loader)):
            pano_label = pano_label.numpy().astype(np.uint32)[0]
            t0 = time.time()
            with torch.cuda.amp.autocast():
                # torch.Size([4, 20, 123174, 1])  torch.Size([4, 123174, 3])  torch.Size([4, 123174, 1])
                pred_sem, pred_offset, pred_hmap = model.infer(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(), pcds_sphere_coord.squeeze(0).cuda())
            t1 = time.time()
            # print('t total epasped:', t1-t0)
            # pdb.set_trace()
            pred_sem = F.softmax(pred_sem, dim=1).mean(dim=0).permute(2, 1, 0).contiguous()[0]
            pred_offset = merge_offset_tta(pred_offset)
            pred_hmap = pred_hmap.mean(dim=0).squeeze(1)

            # make result
            torch.cuda.synchronize()
            t2 = time.time()
            pred_obj_center, pred_panoptic = pv_nms(pcds_xyzi[0, 0, :3, :, 0].T.contiguous().cuda(), pred_sem, pred_offset, pred_hmap)
            torch.cuda.synchronize()
            t3 = time.time()
            # print('post:', t3-t2)
            pred_panoptic = pred_panoptic.cpu().numpy().astype(np.uint32)
            # pdb.set_trace()
            if save_results:
                frame_id = fn[0].split('.')[0]
                pre_dir = os.path.join("./experiments/config_mvfcev2ctx_sgd_wce_fp32_lossv2_single_newcpaug/sequences/08/predictions", '{:0>6}.label'.format(frame_id))
                # np.save(pre_dir, pred_panoptic ) # invalid, the number of read file is lager than the number of write(original)
                pred_panoptic.tofile(pre_dir)
                # pdb.set_trace()
            with open('datasets/semantic-kitti.yaml', 'r') as f:
                kitti_cfg = yaml.load(f)
                V.draw_scenes(pcds_xyzi[0][0].squeeze(2).permute(1,0), pred_panoptic, kitti_cfg, category_list)

            criterion_pano.addBatch(pred_panoptic & 0xFFFF, pred_panoptic, pano_label & 0xFFFF, pano_label)
        
        metric = criterion_pano.get_metric()
        string = 'Epoch {}'.format(epoch)
        for key in metric:
            string = string + '; ' + key + ': ' + str(metric[key])
        f.write(string + '\n')
    f.close()


def val(epoch, model, val_loader, category_list, save_path, rank=0):
    criterion_pano = PanopticEval(category_list, None, [0], min_points=50)
    model.eval()
    pv_nms = pytorch_lib.PointVoteNMS(model.point_nms_dic)
    f = open(os.path.join(save_path, 'record_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset, pano_label, seq_id, fn) in tqdm.tqdm(enumerate(val_loader)):
            pano_label = pano_label.numpy().astype(np.uint32)[0]
            pred_sem, pred_offset, pred_hmap = model.infer(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(), pcds_sphere_coord.squeeze(0).cuda())

            pred_sem = F.softmax(pred_sem, dim=1).mean(dim=0).permute(2, 1, 0).contiguous()[0]
            pred_offset = merge_offset_tta(pred_offset)
            pred_hmap = pred_hmap.mean(dim=0).squeeze(1)

            # make result
            pred_obj_center, pred_panoptic = pv_nms(pcds_xyzi[0, 0, :3, :, 0].T.contiguous().cuda(), pred_sem, pred_offset, pred_hmap)
            pred_panoptic = pred_panoptic.cpu().numpy().astype(np.uint32)

            criterion_pano.addBatch(pred_panoptic & 0xFFFF, pred_panoptic, pano_label & 0xFFFF, pano_label)
        
        metric = criterion_pano.get_metric()
        string = 'Epoch {}'.format(epoch)
        for key in metric:
            string = string + '; ' + key + ': ' + str(metric[key])
        f.write(string + '\n')
    f.close()


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()
    # pdb.set_trace()

    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    # reset dist
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # define dataloader
    val_dataset = eval('datasets.{}.DataloadVal'.format(pDataset.Val.data_src))(pDataset.Val)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=pDataset.Val.num_workers,
                            pin_memory=True)

    # define model
    model = eval(pModel.prefix).AttNet(pModel)
    model.cuda()
    model.eval()

    for epoch in range(args.start_epoch, args.end_epoch + 1, world_size):
        # pdb.set_trace()
        if (epoch + rank) < (args.end_epoch + 1):
            pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(epoch + rank))
            model.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
            # pdb.set_trace()
            if pGen.fp16:
                val_fp16(epoch + rank, model, val_loader, pGen.category_list, save_path, rank)
            else:
                val(epoch + rank, model, val_loader, pGen.category_list, save_path, rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=0)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)