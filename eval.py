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
from dataset.eval_dataset_nocs import Dataset
from libs.network import KeyNet
from libs.loss import Loss
import copy

choose_cate_list = [1,2,3,4,5,6]
resume_models = ['model_112_0.184814792342484_bottle.pth',
                 'model_120_0.10268432162888348_bowl.pth',
                 'model_118_0.2008235973417759_camera.pth', 
                 'model_107_0.18291547849029302_can.pth',
                 'model_117_0.12762234719470145_laptop.pth',
                 'model_102_0.1468337191492319_mug.pth']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = 'My_NOCS', help='dataset root dir')
parser.add_argument('--eval_id', type=int, default = 1, help='the evaluation id')
parser.add_argument('--ite', type=int, default=10, help='first frame fix iteration')
parser.add_argument('--num_kp', type=int, default = 8, help='num of kp')
parser.add_argument('--num_points', type=int, default = 500, help='num of input points')
parser.add_argument('--num_cates', type=int, default = 6, help='number of categories')
parser.add_argument('--outf', type=str, default = 'models/', help='load model dir')
opt = parser.parse_args()

if not os.path.exists('eval_results'):
    os.makedirs('eval_results')

if not os.path.exists('eval_results/TEST_{0}'.format(opt.eval_id)):
    os.makedirs('eval_results/TEST_{0}'.format(opt.eval_id))
    for item in choose_cate_list:
        os.makedirs('eval_results/TEST_{0}/temp_{1}'.format(opt.eval_id, item))

for choose_cate in choose_cate_list:
    model = KeyNet(num_points = opt.num_points, num_key = opt.num_kp, num_cates = opt.num_cates)
    model.cuda()
    model.eval()

    model.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, resume_models[choose_cate-1])))

    pconf = torch.ones(opt.num_kp) / opt.num_kp
    pconf = Variable(pconf).cuda()

    test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, choose_cate, 1000)
    criterion = Loss(opt.num_kp, opt.num_cates)

    eval_list_file = open('dataset/eval_list/eval_list_{0}.txt'.format(choose_cate), 'r')
    while 1:
        input_line = eval_list_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        _, choose_obj, choose_video = input_line.split(' ')

        try:
            current_r, current_t = test_dataset.getfirst(choose_obj, choose_video)
            rad_t = np.array([random.uniform(-0.02, 0.02) for i in range(3)]) * 1000.0
            current_t += rad_t

            if opt.ite != 0:
                min_dis = 1000.0
                for iterative in range(opt.ite):  
                    img_fr, choose_fr, cloud_fr, anchor, scale = test_dataset.getone(current_r, current_t)
                    img_fr, choose_fr, cloud_fr, anchor, scale = Variable(img_fr).cuda(), \
                                                         Variable(choose_fr).cuda(), \
                                                         Variable(cloud_fr).cuda(), \
                                                         Variable(anchor).cuda(), \
                                                         Variable(scale).cuda()
                    Kp_fr, att_fr = model.eval_forward(img_fr, choose_fr, cloud_fr, anchor, scale, 0.0, True)
                    new_t, att, kp_dis = criterion.ev_zero(Kp_fr[0], att_fr[0])

                    if min_dis > kp_dis:
                        min_dis = kp_dis
                        best_current_r = copy.deepcopy(current_r)
                        best_current_t = copy.deepcopy(current_t)
                        best_att = copy.deepcopy(att)
                        print(min_dis)

                    current_t = current_t + np.dot(new_t, current_r.T)
                current_r, current_t, att = best_current_r, best_current_t, best_att

            img_fr, choose_fr, cloud_fr, anchor, scale = test_dataset.getone(current_r, current_t)
            img_fr, choose_fr, cloud_fr, anchor, scale = Variable(img_fr).cuda(), \
                                                 Variable(choose_fr).cuda(), \
                                                 Variable(cloud_fr).cuda(), \
                                                 Variable(anchor).cuda(), \
                                                 Variable(scale).cuda()
            Kp_fr, att_fr = model.eval_forward(img_fr, choose_fr, cloud_fr, anchor, scale, 0.0, True)

            test_dataset.projection('eval_results/TEST_{0}/temp_{1}/{2}_{3}'.format(opt.eval_id, choose_cate, choose_obj, choose_video), Kp_fr[0], current_r, current_t, scale, att_fr[0], True, 0.0)

            min_dis = 0.0005
            while 1:
                img_fr, choose_fr, cloud_fr, anchor, scale = test_dataset.getone(current_r, current_t)
                img_fr, choose_fr, cloud_fr, anchor, scale = Variable(img_fr).cuda(), \
                                                     Variable(choose_fr).cuda(), \
                                                     Variable(cloud_fr).cuda(), \
                                                     Variable(anchor).cuda(), \
                                                     Variable(scale).cuda()
                Kp_to, att_to = model.eval_forward(img_fr, choose_fr, cloud_fr, anchor, scale, min_dis, False)

                min_dis = 1000.0
                lenggth = len(Kp_to)
                for idx in range(lenggth):
                    Kp_real, new_r, new_t, kp_dis, att = criterion.ev(Kp_fr[0], Kp_to[idx], att_to[idx])

                    if min_dis > kp_dis:
                        best_kp = Kp_to[idx]
                        min_dis = kp_dis
                        best_r = new_r
                        best_t = new_t
                        best_att = copy.deepcopy(att)
                print(min_dis)

                current_t = current_t + np.dot(best_t, current_r.T)
                current_r = np.dot(current_r, best_r)

                test_dataset.projection('eval_results/TEST_{0}/temp_{1}/{2}_{3}'.format(opt.eval_id, choose_cate, choose_obj, choose_video), Kp_real, current_r, current_t, scale, best_att, True, min_dis)

                print("NEXT FRAME!!!")

        except:
            continue
    print("FINISH!!!")
