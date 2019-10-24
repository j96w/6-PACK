from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import math
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from libs.knn.__init__ import KNearestNeighbor
import torch.distributions as tdist
import copy

class Loss(_Loss):
    def __init__(self, num_key, num_cate):
        super(Loss, self).__init__(True)
        self.num_key = num_key
        self.num_cate = num_cate

        self.oneone = Variable(torch.ones(1)).cuda()

        self.normal = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.0005]))

        self.pconf = torch.ones(num_key) / num_key
        self.pconf = Variable(self.pconf).cuda()

        self.sym_axis = Variable(torch.from_numpy(np.array([0, 1, 0]).astype(np.float32))).cuda().view(1, 3, 1)
        self.threezero = Variable(torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))).cuda()

        self.zeros = torch.FloatTensor([0.0 for j in range(num_key-1) for i in range(num_key)]).cuda()

        self.select1 = torch.tensor([i for j in range(num_key-1) for i in range(num_key)]).cuda()
        self.select2 = torch.tensor([(i%num_key) for j in range(1, num_key) for i in range(j, j+num_key)]).cuda()

        self.knn = KNearestNeighbor(1)

    def estimate_rotation(self, pt0, pt1, sym_or_not):
        pconf2 = self.pconf.view(1, self.num_key, 1)
        cent0 = torch.sum(pt0 * pconf2, dim=1).repeat(1, self.num_key, 1).contiguous()
        cent1 = torch.sum(pt1 * pconf2, dim=1).repeat(1, self.num_key, 1).contiguous()

        diag_mat = torch.diag(self.pconf).unsqueeze(0)
        x = (pt0 - cent0).transpose(2, 1).contiguous()
        y = pt1 - cent1

        pred_t = cent1 - cent0

        cov = torch.bmm(torch.bmm(x, diag_mat), y).contiguous().squeeze(0)

        u, _, v = torch.svd(cov)

        u = u.transpose(1, 0).contiguous()
        d = torch.det(torch.mm(v, u)).contiguous().view(1, 1, 1).contiguous()
        u = u.transpose(1, 0).contiguous().unsqueeze(0)

        ud = torch.cat((u[:, :, :-1], u[:, :, -1:] * d), dim=2)
        v = v.transpose(1, 0).contiguous().unsqueeze(0)

        pred_r = torch.bmm(ud, v).transpose(2, 1).contiguous()

        if sym_or_not:
            pred_r = torch.bmm(pred_r, self.sym_axis).contiguous().view(-1).contiguous()

        return pred_r

    def estimate_pose(self, pt0, pt1):
        pconf2 = self.pconf.view(1, self.num_key, 1)
        cent0 = torch.sum(pt0 * pconf2, dim=1).repeat(1, self.num_key, 1).contiguous()
        cent1 = torch.sum(pt1 * pconf2, dim=1).repeat(1, self.num_key, 1).contiguous()

        diag_mat = torch.diag(self.pconf).unsqueeze(0)
        x = (pt0 - cent0).transpose(2, 1).contiguous()
        y = pt1 - cent1

        pred_t = cent1 - cent0

        cov = torch.bmm(torch.bmm(x, diag_mat), y).contiguous().squeeze(0)

        u, _, v = torch.svd(cov)

        u = u.transpose(1, 0).contiguous()
        d = torch.det(torch.mm(v, u)).contiguous().view(1, 1, 1).contiguous()
        u = u.transpose(1, 0).contiguous().unsqueeze(0)

        ud = torch.cat((u[:, :, :-1], u[:, :, -1:] * d), dim=2)
        v = v.transpose(1, 0).contiguous().unsqueeze(0)

        pred_r = torch.bmm(ud, v).transpose(2, 1).contiguous()
        return pred_r, pred_t[:, 0, :].view(1, 3)

    def change_to_ver(self, Kp):
        pconf2 = self.pconf.view(1, self.num_key, 1)
        cent0 = torch.sum(Kp * pconf2, dim=1).view(-1).contiguous()
        
        num_kp = self.num_key
        ver_Kp_1 = Kp[:, :, 1].view(1, num_kp, 1).contiguous()

        kk_1 = Kp[:, :, 0].view(1, num_kp, 1).contiguous()
        kk_2 = Kp[:, :, 2].view(1, num_kp, 1).contiguous()
        rad = torch.cat((kk_1, kk_2), dim=2).contiguous()
        ver_Kp_2 = torch.norm(rad, dim=2).view(1, num_kp, 1).contiguous()

        tmp_aim_0 = torch.cat((Kp[:, 1:, :], Kp[:, 0:1, :]), dim=1).contiguous()
        aim_0_x = tmp_aim_0[:, :, 0].view(-1).contiguous()
        aim_0_y = tmp_aim_0[:, :, 2].view(-1).contiguous()

        aim_1_x = Kp[:, :, 0].view(-1).contiguous()
        aim_1_y = Kp[:, :, 2].view(-1).contiguous()

        angle = torch.atan2(aim_1_y, aim_1_x) - torch.atan2(aim_0_y, aim_0_x)
        angle[angle < 0] += 2 * math.pi
        ver_Kp_3 = angle.view(1, num_kp, 1).contiguous() * 0.01

        ver_Kp = torch.cat((ver_Kp_1, ver_Kp_2, ver_Kp_3), dim=2).contiguous()

        return ver_Kp, cent0

    def forward(self, Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, scale, cate):
        if cate.view(-1).item() in [2, 4, 5]:
            sym_or_not = False
        else:
            sym_or_not = True

        num_kp = self.num_key
        num_anc = len(anc_fr[0])


        ############ Attention Loss
        gt_t_fr = t_fr.view(1, 1, 3).repeat(1, num_anc, 1)
        min_fr = torch.min(torch.norm(anc_fr - gt_t_fr, dim=2).view(-1))
        loss_att_fr = torch.sum(((torch.norm(anc_fr - gt_t_fr, dim=2).view(1, num_anc) - min_fr) * att_fr).contiguous().view(-1))

        gt_t_to = t_to.view(1, 1, 3).repeat(1, num_anc, 1)
        min_to = torch.min(torch.norm(anc_to - gt_t_to, dim=2).view(-1))
        loss_att_to = torch.sum(((torch.norm(anc_to - gt_t_to, dim=2).view(1, num_anc) - min_to) * att_to).contiguous().view(-1))

        loss_att = (loss_att_fr + loss_att_to).contiguous() / 2.0

        ############# Different View Loss
        gt_Kp_fr = torch.bmm(Kp_fr - t_fr, r_fr).contiguous()
        gt_Kp_to = torch.bmm(Kp_to - t_to, r_to).contiguous()

        if sym_or_not:
            ver_Kp_fr, cent_fr = self.change_to_ver(gt_Kp_fr)
            ver_Kp_to, cent_to = self.change_to_ver(gt_Kp_to)
            Kp_dis = torch.mean(torch.norm((ver_Kp_fr - ver_Kp_to), dim=2), dim=1)
            Kp_cent_dis = (torch.norm(cent_fr - self.threezero) + torch.norm(cent_to - self.threezero)) / 2.0
        else:
            Kp_dis = torch.mean(torch.norm((gt_Kp_fr - gt_Kp_to), dim=2), dim=1)
            cent_fr = torch.mean(gt_Kp_fr, dim=1).view(-1).contiguous()
            cent_to = torch.mean(gt_Kp_to, dim=1).view(-1).contiguous()
            Kp_cent_dis = (torch.norm(cent_fr - self.threezero) + torch.norm(cent_to - self.threezero)) / 2.0


        ############# Pose Error Loss
        rot_Kp_fr = (Kp_fr - t_fr).contiguous()
        rot_Kp_to = (Kp_to - t_to).contiguous()
        rot = torch.bmm(r_to, r_fr.transpose(2, 1))

        if sym_or_not:
            rot = torch.bmm(rot, self.sym_axis).view(-1)
            pred_r = self.estimate_rotation(rot_Kp_fr, rot_Kp_to, sym_or_not)
            loss_rot = (torch.acos(torch.sum(pred_r * rot) / (torch.norm(pred_r) * torch.norm(rot)))).contiguous()
            loss_rot = loss_rot
        else:
            pred_r = self.estimate_rotation(rot_Kp_fr, rot_Kp_to, sym_or_not)
            frob_sqr = torch.sum(((pred_r - rot) * (pred_r - rot)).view(-1)).contiguous()
            frob = torch.sqrt(frob_sqr).unsqueeze(0).contiguous()
            cc = torch.cat([self.oneone, frob / (2 * math.sqrt(2))]).contiguous()
            loss_rot = 2.0 * torch.mean(torch.asin(torch.min(cc))).contiguous()


        ############# Close To Surface Loss
        bs = 1
        num_p = 1
        num_point_mesh = self.num_key

        target_fr = mesh[0].transpose(1, 0).contiguous().view(3, -1)
        pred_fr = gt_Kp_fr.permute(2, 0, 1).contiguous().view(3, -1)
        inds = self.knn(target_fr.unsqueeze(0), pred_fr.unsqueeze(0))
        target_fr = torch.index_select(target_fr, 1, inds.view(-1) - 1)
        target_fr = target_fr.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
        pred_fr = pred_fr.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

        loss_surf_fr = torch.mean(torch.norm((pred_fr - target_fr), dim=2), dim=1)

        target_to = mesh[0].transpose(1, 0).contiguous().view(3, -1)
        pred_to = gt_Kp_to.permute(2, 0, 1).contiguous().view(3, -1)
        inds = self.knn(target_to.unsqueeze(0), pred_to.unsqueeze(0))
        target_to = torch.index_select(target_to, 1, inds.view(-1) - 1)
        target_to = target_to.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
        pred_to = pred_to.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

        loss_surf_to = torch.mean(torch.norm((pred_to - target_to), dim=2), dim=1)

        loss_surf = (loss_surf_fr + loss_surf_to).contiguous() / 2.0


        ############# Separate Loss
        scale = scale.view(-1)
        max_rad = torch.norm(scale).item()

        gt_Kp_fr_select1 = torch.index_select(gt_Kp_fr, 1, self.select1).contiguous()
        gt_Kp_fr_select2 = torch.index_select(gt_Kp_fr, 1, self.select2).contiguous()
        loss_sep_fr = torch.norm((gt_Kp_fr_select1 - gt_Kp_fr_select2), dim=2).view(-1).contiguous()
        loss_sep_fr = torch.max(self.zeros, max_rad/8.0 - loss_sep_fr).contiguous()
        loss_sep_fr = torch.mean(loss_sep_fr).contiguous()

        gt_Kp_to_select1 = torch.index_select(gt_Kp_to, 1, self.select1).contiguous()
        gt_Kp_to_select2 = torch.index_select(gt_Kp_to, 1, self.select2).contiguous()
        loss_sep_to = torch.norm((gt_Kp_to_select1 - gt_Kp_to_select2), dim=2).view(-1).contiguous()
        loss_sep_to = torch.max(self.zeros, max_rad/8.0 - loss_sep_to).contiguous()
        loss_sep_to = torch.mean(loss_sep_to).contiguous()

        loss_sep = (loss_sep_fr + loss_sep_to) / 2.0

        ########### SUM UP

        loss = loss_att * 4.0 + Kp_dis * 3.0 + Kp_cent_dis + loss_rot * 0.2 + loss_surf * 3.0 + loss_sep
        score = (loss_att * 4.0 + Kp_dis * 3.0 + Kp_cent_dis + loss_rot * 0.2).item()
        print(cate.view(-1).item(), loss_att.item(), Kp_dis.item(), Kp_cent_dis.item(), loss_rot.item(), loss_surf.item(), loss_sep)

        return loss, score


    def ev(self, Kp_fr, Kp_to, att_to):
        ori_Kp_fr = Kp_fr
        ori_Kp_to = Kp_to

        new_r, new_t = self.estimate_pose(Kp_fr, Kp_to)

        Kp_to = torch.bmm((ori_Kp_to - new_t), new_r)

        Kp_dis = torch.mean(torch.norm((Kp_fr - Kp_to), dim=2), dim=1)

        new_t *= 1000.0
        return ori_Kp_fr, new_r.detach().cpu().numpy()[0], new_t.detach().cpu().numpy()[0], Kp_dis.item(), att_to

    def ev_zero(self, Kp_fr, att_fr):
        pconf2 = self.pconf.view(1, self.num_key, 1)
        new_t = torch.sum(Kp_fr * pconf2, dim=1).view(1, 3).contiguous()

        kp_dis = torch.norm(new_t.view(-1))

        new_t *= 1000.0
        return new_t.detach().cpu().numpy()[0], att_fr, kp_dis.item()

    def inf(self, Kp_fr, Kp_to):
        ori_Kp_to = Kp_to

        new_r, new_t = self.estimate_pose(Kp_fr, Kp_to)

        Kp_to = torch.bmm((ori_Kp_to - new_t), new_r)

        Kp_dis = torch.mean(torch.norm((Kp_fr - Kp_to), dim=2), dim=1)

        new_t *= 1000.0
        return new_r.detach().cpu().numpy()[0], new_t.detach().cpu().numpy()[0], Kp_dis.item()

    def inf_zero(self, Kp_fr):
        new_t = torch.sum(Kp_fr * self.pconf2, dim=1).view(1, 3).contiguous()

        Kp_dis = torch.norm(new_t.view(-1))

        new_t *= 1000.0
        return new_t.detach().cpu().numpy()[0], Kp_dis.item()
