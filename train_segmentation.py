from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default = '',  help='model path')
# parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
# parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
# parser.add_argument('--nclass', type=int, default=0, help='number of classes')
# parser.add_argument('--outf', type=str, default='seg',  help='output folder')
# parser.add_argument('--model', type=str, default = '',  help='model path')
# parser.add_argument('--root', type=str, default = 'shapenetcore_partanno_segmentation_benchmark_v0',  help='model path')
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

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

print('preparing dataset')
dataset = PartDataset(root = cfg['dataset_root'], classification = cfg['classification'], class_choice = [cfg['chosen_class']], \
                    npoints = cfg['num_points'], nclasses = cfg['num_seg_classes'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size = cfg['batch_size'],
                                        shuffle = cfg['shuffle'], num_workers = cfg['workers'])

test_dataset = PartDataset(root = cfg['dataset_root'], classification = cfg['classification'], class_choice = [cfg['chosen_class']], \
                        npoints = cfg['num_points'], nclasses = cfg['num_seg_classes'])
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size = cfg['batch_size'],
                                            shuffle = cfg['shuffle'], num_workers = cfg['workers'])

# print(len(dataset), len(test_dataset))
print(f"Training on {len(dataset)} samples..")

if cfg['debug_vis'] == True:
    import open3d as o3d 
    import matplotlib
    point, seg = dataset[np.round(np.random.uniform(len(dataset))).astype(int)]
    point_np = point.numpy()
    pcl = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_np))
    cmap = matplotlib.colormaps['jet'].resampled(cfg['num_seg_classes'])
    colors = cmap(seg.numpy()-1)[:,:3]
    pcl.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcl])
    breakpoint()

num_classes = cfg['num_seg_classes']
print('classes', num_classes)

output_dir = os.path.join(os.getcwd(), cfg['models_path'])
os.makedirs(output_dir, exist_ok=True)

blue = lambda x:'\033[94m' + x + '\033[0m'

segNet = PointNetDenseCls(k = num_classes, num_points = cfg['num_points'])

if cfg['continue_training'] == True:
    if opt.model != '':
        segNet.load_state_dict(torch.load(cfg['continue_training']['ckp_path']))

if cfg['optimizer'] == 'SGD':
    optimizer = optim.SGD(segNet.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
elif cfg['optimizer'] == 'Adam':
    optimizer = optim.Adam(segNet.parameters(), lr=cfg['lr'])
else:
    optimizer = optim.Adagrad(segNet.parameters(), lr=cfg['lr'])
print("using as optimizer", optimizer)

if torch.cuda.is_available():
    segNet.cuda()

num_batch = len(dataset) / cfg['batch_size']

loss_history = []

print("Weights")
weight = np.ones(num_classes, dtype=np.float32)
for w in range(1, num_classes):
    #weight[w] = 1 / (num_classes - 1)
    weight[w] += cfg['scaling_factor_fragments']
weight /= np.sum(weight) 
print(weight)


print('starting training..')
for epoch in range(cfg['epochs']):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points, target = Variable(points), Variable(target)
        points = points.transpose(2,1)
        if torch.cuda.is_available():
            points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        segNet = segNet.train()
        pred, _ = segNet(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        #pred_labels = torch.argmax(pred, axis=1)
        target_oh = nn.functional.one_hot(target, num_classes=num_classes)
        loss = F.cross_entropy(pred, target_oh, weight=torch.from_numpy(weight))
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(), correct.item()/float(cfg['batch_size'] * cfg['num_points'])))
        loss_history.append(loss.item())
        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2,1)
            if torch.cuda.is_available():
                points, target = points.cuda(), target.cuda()
            segNet = segNet.eval()
            pred, _ = segNet(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1,1)[:,0] - 1
            target_oh = nn.functional.one_hot(target, num_classes=num_classes).double()
            loss = F.cross_entropy(pred, target_oh, weight=torch.from_numpy(weight))
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(cfg['batch_size'] * cfg['num_points'])))

    torch.save(segNet.state_dict(), '%s/seg_model_%depochs_weighted_CE.pth' % (output_dir, epoch))

import matplotlib.pyplot as plt 
plt.plot(loss_history)
plt.show()
breakpoint()
np.savetxt(os.path.join(output_dir, 'weighted_loss_CE.txt'), loss_history)