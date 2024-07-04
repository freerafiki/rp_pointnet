from __future__ import print_function
#from show3d_balls import *
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d 
import matplotlib
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default = '',  help='model path')
opt = parser.parse_args()
print (opt)

with open('config.yaml', 'r') as yf:
    cfg = yaml.safe_load(yf)

print()
print("#" * 60)
print("# Parameters")
for ck in cfg.keys():
    print(f"# {ck}: {cfg[ck]}")
print("#" * 60)
print()

test_dataset = PartDataset(root = cfg['dataset_root'], classification = cfg['classification'], class_choice = [cfg['chosen_class']], \
                        npoints = cfg['num_points'], nclasses = cfg['num_seg_classes'])
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size = cfg['batch_size'],
                                            shuffle = False, num_workers = cfg['workers'])

point, seg = test_dataset[np.round(np.random.uniform(len(test_dataset))).astype(int)]
print(point.size(), seg.size())

point_np = point.numpy()
gt = seg.numpy() - 1
# cmap = plt.cm.get_cmap("hsv", cfg['num_seg_classes'])
# cmap = np.array([cmap(i) for i in range(cfg['num_seg_classes'])])[:,:3]
# gt = cmap[seg.numpy() - 1, :]

trained_segNet = PointNetDenseCls(k = cfg['num_seg_classes'], num_points=cfg['num_points'])
model_path = os.path.join(os.getcwd(), cfg['models_path'], f"seg_model_{cfg['epochs']-1}.pth")
trained_segNet.load_state_dict(torch.load(model_path))
trained_segNet.eval()

point = point.transpose(1,0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _ = trained_segNet(point)
pred_lab = torch.argmax(torch.squeeze(pred), axis=1)
# pred_choice = pred.data.max(2)[1]
cmapjet = matplotlib.colormaps['jet'].resampled(cfg['num_seg_classes'])
pred_in_colors = cmapjet(pred_lab+1)[:,:3]

errors = np.abs(pred_lab - gt)
print("#" * 60)
print("# STATS")
err_np = errors.numpy()
num_err = np.sum(err_np > 0)
num_err_norm = num_err / (err_np.shape[0])
acc = 1 - num_err_norm 
print(f"# acc: {acc:03f}")
ferr_np = (errors.numpy() > 0) * (gt > 1)
fnum_err = np.sum(err_np > 0)
fnum_err_norm = fnum_err / (ferr_np.shape[0])
facc = 1 - fnum_err_norm 
print(f"# acc (fragments): {facc}")
print("#" * 60)

# point, seg = dataset[]
# point_np = point.numpy()
# pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_np))
# cmap = matplotlib.colormaps['jet'].resampled(cfg['num_seg_classes'])
# colors = cmap(seg.numpy()-1)[:,:3]
# pcl.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([pcl])
pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_np))
pcl.colors = o3d.utility.Vector3dVector(pred_in_colors)

o3d.visualization.draw_geometries([pcl])
