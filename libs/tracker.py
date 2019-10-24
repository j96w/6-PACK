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
from dataset.inference_dataset_nocs import Dataset
from libs.network import KeyNet
from libs.loss import Loss
import copy


class tracker():
    def __init__(self, choose_cate):
        self.resume_models = ['model_112_0.184814792342484_bottle.pth',
                              'model_120_0.10268432162888348_bowl.pth',
                              'model_118_0.2008235973417759_camera.pth', 
                              'model_107_0.18291547849029302_can.pth',
                              'model_117_0.12762234719470145_laptop.pth',
                              'model_102_0.1468337191492319_mug.pth']

        self.first_ite = 10
        self.temp_gap = 3
        self.num_kp = 8
        self.num_points = 500
        self.num_cates = 6
        self.outf = 'models'
        self.min_dis = 0.0005

        self.model = KeyNet(num_points = self.num_points, num_key = self.num_kp, num_cates = self.num_cates)
        self.model.cuda()
        self.model.load_state_dict(torch.load('{0}/{1}'.format(self.outf, self.resume_models[choose_cate-1])))
        self.model.eval()

        self.dataprocess = Dataset(self.num_points)
        self.criterion = Loss(self.num_kp, self.num_cates)

        self.temp_dir = []
        for temp_ite in range(self.temp_gap):
            self.temp_dir.append(np.array([0.0, 0.0, 0.0]))

        self.Kp_fr = Variable(torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))).cuda()
        self.Kp_fr.view(1, 1, 3).repeat(1, self.num_kp, 1)


    def init_estimation(self, rgb_dir, depth_dir, current_r, current_t, bbox):
        for temp_ite in range(self.temp_gap):
            self.temp_dir.append(np.array([0.0, 0.0, 0.0]))
        while len(self.temp_dir) > self.temp_gap:
            del self.temp_dir[0]

        # print(rgb_dir, depth_dir)
        self.dataprocess.add_bbox(bbox)

        if self.first_ite != 0:
            min_dis = 1000.0
            for iterative in range(self.first_ite):  
                img_fr, choose_fr, cloud_fr, anchor, scale = self.dataprocess.getone(rgb_dir, depth_dir, current_r, current_t)
                img_fr, choose_fr, cloud_fr, anchor, scale = Variable(img_fr).cuda(), \
                                                             Variable(choose_fr).cuda(), \
                                                             Variable(cloud_fr).cuda(), \
                                                             Variable(anchor).cuda(), \
                                                             Variable(scale).cuda()
                Kp_fr, _ = self.model.eval_forward(img_fr, choose_fr, cloud_fr, anchor, scale, 0.0, True)
                new_t, kp_dis = self.criterion.inf_zero(Kp_fr[0])

                if min_dis > kp_dis:
                    min_dis = kp_dis
                    best_current_r = copy.deepcopy(current_r)
                    best_current_t = copy.deepcopy(current_t)
                    print(min_dis)

                current_t = current_t + np.dot(new_t, current_r.T)
            current_r, current_t = best_current_r, best_current_t

        img_fr, choose_fr, cloud_fr, anchor, scale = self.dataprocess.getone(rgb_dir, depth_dir, current_r, current_t)
        img_fr, choose_fr, cloud_fr, anchor, scale = Variable(img_fr).cuda(), \
                                                     Variable(choose_fr).cuda(), \
                                                     Variable(cloud_fr).cuda(), \
                                                     Variable(anchor).cuda(), \
                                                     Variable(scale).cuda()
        Kp_fr, _ = self.model.eval_forward(img_fr, choose_fr, cloud_fr, anchor, scale, 0.0, True)

        self.Kp_fr = Kp_fr[0]

        self.dataprocess.projection(rgb_dir, current_r, current_t)
        print("NEXT!!!")

        return current_r, current_t

    def next_estimation(self, rgb_dir, depth_dir, current_r, current_t):
        img_fr, choose_fr, cloud_fr, anchor, scale = self.dataprocess.getone(rgb_dir, depth_dir, current_r, current_t)
        img_fr, choose_fr, cloud_fr, anchor, scale = Variable(img_fr).cuda(), \
                                                     Variable(choose_fr).cuda(), \
                                                     Variable(cloud_fr).cuda(), \
                                                     Variable(anchor).cuda(), \
                                                     Variable(scale).cuda()
        Kp_to, _ = self.model.eval_forward(img_fr, choose_fr, cloud_fr, anchor, scale, 0.0, False)

        self.min_dis = 1000.0
        lenggth = len(Kp_to)
        for idx in range(lenggth):
            new_r, new_t, kp_dis = self.criterion.inf(self.Kp_fr, Kp_to[idx])
            if self.min_dis > kp_dis:
                self.min_dis = kp_dis
                best_r = new_r
                best_t = new_t
        print(self.min_dis)

        current_t = current_t + np.dot(best_t, current_r.T)
        current_r = np.dot(current_r, best_r)

        self.temp_dir.append(copy.deepcopy(best_t / 1000.0))
        if len(self.temp_dir) > self.temp_gap:
            del self.temp_dir[0]

        self.dataprocess.projection(rgb_dir, current_r, current_t)
        print("NEXT!!!")

        return current_r, current_t
