import argparse
import os
import random
import time
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
from dataset.dataset_nocs import Dataset
from libs.network import KeyNet
from libs.loss import Loss

cate_list = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = 'My_NOCS', help='dataset root dir')
parser.add_argument('--resume', type=str, default = '',  help='resume model')
parser.add_argument('--category', type=int, default = 5,  help='category to train')
parser.add_argument('--num_points', type=int, default = 500, help='points')
parser.add_argument('--num_cates', type=int, default = 6, help='number of categories')
parser.add_argument('--workers', type=int, default = 5, help='number of data loading workers')
parser.add_argument('--num_kp', type=int, default = 8, help='number of kp')
parser.add_argument('--outf', type=str, default = 'models/', help='save dir')
parser.add_argument('--lr', default=0.0001, help='learning rate')
opt = parser.parse_args()

model = KeyNet(num_points = opt.num_points, num_key = opt.num_kp, num_cates = opt.num_cates)
model.cuda()

if opt.resume != '':
    model.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume)))

dataset = Dataset('train', opt.dataset_root, True, opt.num_points, opt.num_cates, 5000, opt.category)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, opt.num_cates, 1000, opt.category)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

criterion = Loss(opt.num_kp, opt.num_cates)

best_test = np.Inf
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

for epoch in range(0, 500):
    model.train()
    train_dis_avg = 0.0
    train_count = 0

    optimizer.zero_grad()

    for i, data in enumerate(dataloader, 0):
        img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = data
        img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = Variable(img_fr).cuda(), \
                                                                                                                     Variable(choose_fr).cuda(), \
                                                                                                                     Variable(cloud_fr).cuda(), \
                                                                                                                     Variable(r_fr).cuda(), \
                                                                                                                     Variable(t_fr).cuda(), \
                                                                                                                     Variable(img_to).cuda(), \
                                                                                                                     Variable(choose_to).cuda(), \
                                                                                                                     Variable(cloud_to).cuda(), \
                                                                                                                     Variable(r_to).cuda(), \
                                                                                                                     Variable(t_to).cuda(), \
                                                                                                                     Variable(mesh).cuda(), \
                                                                                                                     Variable(anchor).cuda(), \
                                                                                                                     Variable(scale).cuda(), \
                                                                                                                     Variable(cate).cuda()

        Kp_fr, anc_fr, att_fr = model(img_fr, choose_fr, cloud_fr, anchor, scale, cate, t_fr)
        Kp_to, anc_to, att_to = model(img_to, choose_to, cloud_to, anchor, scale, cate, t_to)

        loss, _ = criterion(Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, scale, cate)
        loss.backward()

        train_dis_avg += loss.item()
        train_count += 1

        if train_count != 0 and train_count % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(train_count, float(train_dis_avg) / 8.0)
            train_dis_avg = 0.0

        if train_count != 0 and train_count % 100 == 0:
            torch.save(model.state_dict(), '{0}/model_current_{1}.pth'.format(opt.outf, cate_list[opt.category-1]))


    optimizer.zero_grad()
    model.eval()
    score = []
    for j, data in enumerate(testdataloader, 0):
        img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = data
        img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = Variable(img_fr).cuda(), \
                                                                                                                     Variable(choose_fr).cuda(), \
                                                                                                                     Variable(cloud_fr).cuda(), \
                                                                                                                     Variable(r_fr).cuda(), \
                                                                                                                     Variable(t_fr).cuda(), \
                                                                                                                     Variable(img_to).cuda(), \
                                                                                                                     Variable(choose_to).cuda(), \
                                                                                                                     Variable(cloud_to).cuda(), \
                                                                                                                     Variable(r_to).cuda(), \
                                                                                                                     Variable(t_to).cuda(), \
                                                                                                                     Variable(mesh).cuda(), \
                                                                                                                     Variable(anchor).cuda(), \
                                                                                                                     Variable(scale).cuda(), \
                                                                                                                     Variable(cate).cuda()

        Kp_fr, anc_fr, att_fr = model(img_fr, choose_fr, cloud_fr, anchor, scale, cate, t_fr)
        Kp_to, anc_to, att_to = model(img_to, choose_to, cloud_to, anchor, scale, cate, t_to)

        _, item_score = criterion(Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, scale, cate)
        
        print(item_score)
        score.append(item_score)

    test_dis = np.mean(np.array(score))
    if test_dis < best_test:
        best_test = test_dis
        torch.save(model.state_dict(), '{0}/model_{1}_{2}_{3}.pth'.format(opt.outf, epoch, test_dis, cate_list[opt.category-1]))
        print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
