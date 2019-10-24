import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from libs.pspnet import PSPNet
import torch.distributions as tdist
import copy

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 256, 1)

        self.all_conv1 = torch.nn.Conv1d(640, 320, 1)
        self.all_conv2 = torch.nn.Conv1d(320, 160, 1)

        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = torch.cat([pointfeat_1, pointfeat_2, x], dim=1).contiguous() #128 + 256 + 256

        x = F.leaky_relu(self.all_conv1(x))
        x = self.all_conv2(x)

        return x


class KeyNet(nn.Module):
    def __init__(self, num_points, num_key, num_cates):
        super(KeyNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)
        self.num_cates = num_cates

        self.sm = torch.nn.Softmax(dim=2)
        
        self.kp_1 = torch.nn.Conv1d(160, 90, 1)
        self.kp_2 = torch.nn.Conv1d(90, 3*num_key, 1)

        self.att_1 = torch.nn.Conv1d(160, 90, 1)
        self.att_2 = torch.nn.Conv1d(90, 1, 1)

        self.sm2 = torch.nn.Softmax(dim=1)

        self.num_key = num_key

        self.threezero = Variable(torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))).cuda().view(1, 1, 3).repeat(1, self.num_points, 1)

    def forward(self, img, choose, x, anchor, scale, cate, gt_t):
        num_anc = len(anchor[0])
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        emb = emb.repeat(1, 1, num_anc).contiguous()

        output_anchor = anchor.view(1, num_anc, 3)
        anchor_for_key = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_key, 1)
        anchor = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_points, 1)
        x = x.view(1, 1, self.num_points, 3).repeat(1, num_anc, 1, 1)
        x = (x - anchor).view(1, num_anc * self.num_points, 3).contiguous()

        x = x.transpose(2, 1).contiguous()
        feat_x = self.feat(x, emb)
        feat_x = feat_x.transpose(2, 1).contiguous()
        feat_x = feat_x.view(1, num_anc, self.num_points, 160).contiguous()

        loc = x.transpose(2, 1).contiguous().view(1, num_anc, self.num_points, 3)
        weight = self.sm(-1.0 * torch.norm(loc, dim=3)).contiguous()
        weight = weight.view(1, num_anc, self.num_points, 1).repeat(1, 1, 1, 160).contiguous()

        feat_x = torch.sum((feat_x * weight), dim=2).contiguous().view(1, num_anc, 160)
        feat_x = feat_x.transpose(2, 1).contiguous()

        kp_feat = F.leaky_relu(self.kp_1(feat_x))
        kp_feat = self.kp_2(kp_feat)
        kp_feat = kp_feat.transpose(2, 1).contiguous()
        kp_x = kp_feat.view(1, num_anc, self.num_key, 3).contiguous()
        kp_x = (kp_x + anchor_for_key).contiguous()

        att_feat = F.leaky_relu(self.att_1(feat_x))
        att_feat = self.att_2(att_feat)
        att_feat = att_feat.transpose(2, 1).contiguous()
        att_feat = att_feat.view(1, num_anc).contiguous()
        att_x = self.sm2(att_feat).contiguous()

        scale_anc = scale.view(1, 1, 3).repeat(1, num_anc, 1)
        output_anchor = (output_anchor * scale_anc).contiguous()
        min_choose = torch.argmin(torch.norm(output_anchor - gt_t, dim=2).view(-1))

        all_kp_x = kp_x.view(1, num_anc, 3*self.num_key).contiguous()
        all_kp_x = all_kp_x[:, min_choose, :].contiguous()
        all_kp_x = all_kp_x.view(1, self.num_key, 3).contiguous()

        scale_kp = scale.view(1, 1, 3).repeat(1, self.num_key, 1)
        all_kp_x = (all_kp_x * scale_kp).contiguous()

        return all_kp_x, output_anchor, att_x

    def eval_forward(self, img, choose, ori_x, anchor, scale, space, first):
        num_anc = len(anchor[0])
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose)
        emb = emb.repeat(1, 1, num_anc).detach()
        #print(emb.size())

        output_anchor = anchor.view(1, num_anc, 3)
        anchor_for_key = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_key, 1)
        anchor = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_points, 1)

        candidate_list = [-10*space, 0.0, 10*space]
        if space != 0.0:
            add_on = []
            for add_x in candidate_list:
                for add_y in candidate_list:
                    for add_z in candidate_list:
                        add_on.append([add_x, add_y, add_z])

            add_on = Variable(torch.from_numpy(np.array(add_on).astype(np.float32))).cuda().view(27, 1, 3)
        else:
            add_on = Variable(torch.from_numpy(np.array([0.0, 0.0, 0.0]).astype(np.float32))).cuda().view(1, 1, 3)

        all_kp_x = []
        all_att_choose = []
        scale_add_on = scale.view(1, 3)

        for tmp_add_on in add_on:
            tmp_add_on_scale = (tmp_add_on / scale_add_on).view(1, 1, 3).repeat(1, self.num_points, 1)
            tmp_add_on_key = (tmp_add_on / scale_add_on).view(1, 1, 3).repeat(1, self.num_key, 1)
            x = ori_x - tmp_add_on_scale

            x = x.view(1, 1, self.num_points, 3).repeat(1, num_anc, 1, 1)
            x = (x - anchor).view(1, num_anc * self.num_points, 3)

            x = x.transpose(2, 1)
            feat_x = self.feat(x, emb)
            feat_x = feat_x.transpose(2, 1)
            feat_x = feat_x.view(1, num_anc, self.num_points, 160).detach()

            loc = x.transpose(2, 1).view(1, num_anc, self.num_points, 3)
            weight = self.sm(-1.0 * torch.norm(loc, dim=3))
            weight = weight.view(1, num_anc, self.num_points, 1).repeat(1, 1, 1, 160)

            feat_x = torch.sum((feat_x * weight), dim=2).view(1, num_anc, 160)
            feat_x = feat_x.transpose(2, 1).detach()

            kp_feat = F.leaky_relu(self.kp_1(feat_x))
            kp_feat = self.kp_2(kp_feat)
            kp_feat = kp_feat.transpose(2, 1)
            kp_x = kp_feat.view(1, num_anc, self.num_key, 3)
            kp_x = (kp_x + anchor_for_key).detach()

            att_feat = F.leaky_relu(self.att_1(feat_x))
            att_feat = self.att_2(att_feat)
            att_feat = att_feat.transpose(2, 1)
            att_feat = att_feat.view(1, num_anc)
            att_x = self.sm2(att_feat).detach()

            if not first:
                att_choose = torch.argmax(att_x.view(-1))
            else:
                att_choose = Variable(torch.from_numpy(np.array([62])).long()).cuda().view(-1)

            scale_anc = scale.view(1, 1, 3).repeat(1, num_anc, 1)
            output_anchor = (output_anchor * scale_anc)

            scale_kp = scale.view(1, 1, 3).repeat(1, self.num_key, 1)
            kp_x = kp_x.view(1, num_anc, 3*self.num_key).detach()
            kp_x = (kp_x[:, att_choose, :].view(1, self.num_key, 3) + tmp_add_on_key).detach()

            kp_x = kp_x * scale_kp

            all_kp_x.append(copy.deepcopy(kp_x.detach()))
            all_att_choose.append(copy.deepcopy(att_choose.detach()))

        return all_kp_x, all_att_choose