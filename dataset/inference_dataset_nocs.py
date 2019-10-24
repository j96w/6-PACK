import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
from transformations import euler_matrix
from torch.autograd import Variable
import argparse
import time
import random
import numpy.ma as ma
import copy
import math
import scipy.misc
import scipy.io as scio
import cv2
from PIL import Image

class Dataset():
    def __init__(self, num_pts):
        self.num_pts = num_pts

        self.cam_cx = 321.24099379
        self.cam_cy = 237.11014479
        self.cam_fx = 537.99040688
        self.cam_fy = 539.09122804
        self.cam_scale = 1.0

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.color = np.array([[255, 69, 0], [124, 252, 0], [0, 238, 238], [238, 238, 0], [155, 48, 255], [0, 0, 238], [255, 131, 250], [189, 183, 107], [165, 42, 42], [0, 234, 0]])

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.bbox = [[0.0, 0.0, 0.0] for k in range(8)]

        self.choose_obj = ''
        self.index = 0
        self.video_id = ''

    def add_bbox(self, bbox):
        self.bbox = np.array(bbox)

    def divide_scale(self, scale, pts):

        pts[:, 0] = pts[:, 0] / scale[0]
        pts[:, 1] = pts[:, 1] / scale[1]
        pts[:, 2] = pts[:, 2] / scale[2]

        return pts

    def get_anchor_box(self, ori_bbox):
        bbox = ori_bbox
        limit = np.array(search_fit(bbox))
        num_per_axis = 5
        gap_max = num_per_axis - 1


        gap_x = (limit[1] - limit[0]) / float(gap_max)
        gap_y = (limit[3] - limit[2]) / float(gap_max)
        gap_z = (limit[5] - limit[4]) / float(gap_max)

        ans = []
        scale = [max(limit[1], -limit[0]), max(limit[3], -limit[2]), max(limit[5], -limit[4])]

        for i in range(0, num_per_axis):
            for j in range(0, num_per_axis):
                for k in range(0, num_per_axis):
                    ans.append([limit[0] + i * gap_x, limit[2] + j * gap_y, limit[4] + k * gap_z])

        ans = np.array(ans)
        scale = np.array(scale)

        ans = self.divide_scale(scale, ans)

        return ans, scale


    def change_to_scale(self, scale, cloud_fr):
        cloud_fr = self.divide_scale(scale, cloud_fr)

        return cloud_fr

    def re_scale(self, target_fr, target_to):
        ans_scale = target_fr / target_to
        ans_target = target_fr

        ans_scale = ans_scale[0][0]

        return ans_target, ans_scale

    def enlarge_bbox(self, target):

        limit = np.array(search_fit(target))
        longest = max(limit[1]-limit[0], limit[3]-limit[2], limit[5]-limit[4])
        longest = longest * 1.3

        scale1 = longest / (limit[1]-limit[0])
        scale2 = longest / (limit[3]-limit[2])
        scale3 = longest / (limit[5]-limit[4])

        target[:, 0] *= scale1
        target[:, 1] *= scale2
        target[:, 2] *= scale3

        return target

    def load_depth(self, depth_path):
        depth = cv2.imread(depth_path, -1)

        if len(depth.shape) == 3:
            depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2])
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'

        return depth16

    def getone(self, img_dir, depth_dir, current_r, current_t):
        img = Image.open(img_dir)
        depth = np.load(depth_dir)

        target_r = current_r
        target_t = current_t

        cam_cx = self.cam_cx
        cam_cy = self.cam_cy
        cam_fx = self.cam_fx
        cam_fy = self.cam_fy
        cam_scale = self.cam_scale

        target = self.bbox
        target = self.enlarge_bbox(copy.deepcopy(target))

        target_tmp = np.dot(target, target_r.T) + target_t
        target_tmp[:, 0] *= -1.0
        target_tmp[:, 1] *= -1.0
        rmin, rmax, cmin, cmax = get_2dbbox(target_tmp, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale)
        limit = search_fit(target)

        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        img = img / 255.0

        depth = depth[rmin:rmax, cmin:cmax]
        choose = (depth.flatten() > -10000.0).nonzero()[0]

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)

        cloud = np.dot(cloud - target_t, target_r)

        choose_temp = (cloud[:, 0] > limit[0]) * (cloud[:, 0] < limit[1]) * (cloud[:, 1] > limit[2]) * (cloud[:, 1] < limit[3]) * (cloud[:, 2] > limit[4]) * (cloud[:, 2] < limit[5])

        choose = ((depth.flatten() != 0.0) * choose_temp).nonzero()[0]

        if len(choose) == 0:
            choose = np.array([0])
        if len(choose) > self.num_pts:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pts] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pts - len(choose)), 'wrap')

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)
        choose = np.array([choose])

        cloud = np.dot(cloud - target_t, target_r)


        cloud = cloud / 1000.0
        target = target / 1000.0

        anchor_box, scale = self.get_anchor_box(target)
        cloud_fr = self.change_to_scale(scale, cloud)

        return self.norm(torch.from_numpy(img.astype(np.float32))).unsqueeze(0), \
               torch.LongTensor(choose.astype(np.int32)).unsqueeze(0), \
               torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0), \
               torch.from_numpy(anchor_box.astype(np.float32)).unsqueeze(0), \
               torch.from_numpy(scale.astype(np.float32)).unsqueeze(0)

    def build_frame(self, min_x, max_x, min_y, max_y, min_z, max_z):
        bbox = []
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, min_y, min_z])
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, min_y, max_z])
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, max_y, min_z])
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, max_y, max_z])

        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([min_x, i, min_z])
        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([min_x, i, max_z])
        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([max_x, i, min_z])
        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([max_x, i, max_z])

        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([min_x, min_y, i])
        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([min_x, max_y, i])
        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([max_x, min_y, i])
        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([max_x, max_y, i])
        bbox = np.array(bbox)

        return bbox

    def projection(self, img_dir, current_r, current_t):
        img = np.array(Image.open(img_dir))

        cam_cx = self.cam_cx
        cam_cy = self.cam_cy
        cam_fx = self.cam_fx
        cam_fy = self.cam_fy
        cam_scale = self.cam_scale

        target_r = current_r
        target_t = current_t

        target = self.bbox
        limit = search_fit(target)
        bbox = self.build_frame(limit[0], limit[1], limit[2], limit[3], limit[4], limit[5])

        bbox = np.dot(bbox, target_r.T) + target_t
        bbox[:, 0] *= -1.0
        bbox[:, 1] *= -1.0

        fw = open('results/{0}_pose.txt'.format(self.index), 'w')

        for it in target_r:
            fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        it = target_t
        fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        fw.close()

        for tg in bbox:
            y = int(tg[0] * cam_fx / tg[2] + cam_cx)
            x = int(tg[1] * cam_fy / tg[2] + cam_cy)

            if x - 3 < 0 or x + 3 > 479 or y - 3 < 0 or y + 3 > 639:
                continue

            for xxx in range(x-2, x+3):
                for yyy in range(y-2, y+3):
                    img[xxx][yyy] = self.color[1]

        scipy.misc.imsave('results/{0}.png'.format(self.index), img)

        self.index += 1



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_2dbbox(cloud, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale):
    rmin = 10000
    rmax = -10000
    cmin = 10000
    cmax = -10000
    for tg in cloud:
        p1 = int(tg[0] * cam_fx / tg[2] + cam_cx)
        p0 = int(tg[1] * cam_fy / tg[2] + cam_cy)
        if p0 < rmin:
            rmin = p0
        if p0 > rmax:
            rmax = p0
        if p1 < cmin:
            cmin = p1
        if p1 > cmax:
            cmax = p1
    rmax += 1
    cmax += 1
    if rmin < 0:
        rmin = 0
    if cmin < 0:
        cmin = 0
    if rmax >= 480:
        rmax = 479
    if cmax >= 640:
        cmax = 639

    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
        
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
        
    return rmin, rmax, cmin, cmax


def search_fit(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    return [min_x, max_x, min_y, max_y, min_z, max_z]


