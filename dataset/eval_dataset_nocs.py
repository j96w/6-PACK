from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
from libs.transformations import euler_matrix
import argparse
import time
import random
import numpy.ma as ma
import copy
import math
import scipy.misc
import scipy.io as scio
import cv2
import _pickle as cPickle

class Dataset():
    def __init__(self, mode, root, add_noise, num_pts, cate_id, count):
        self.root = root
        self.add_noise = add_noise
        self.mode = mode
        self.cate_id = cate_id
        self.num_pts = num_pts

        self.real_obj_list = {}

        self.real_obj_name_list = os.listdir('{0}/data_list/real_{1}/{2}/'.format(self.root, self.mode, self.cate_id))
        for item in self.real_obj_name_list:
            print(item)
            self.real_obj_list[item] = []

            input_file = open('{0}/data_list/real_{1}/{2}/{3}/list.txt'.format(self.root, self.mode, self.cate_id, item), 'r')

            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.real_obj_list[item].append('{0}/data/{1}'.format(self.root, input_line))
            input_file.close()

        self.mesh = []
        input_file = open('dataset/sphere.xyz', 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            self.mesh.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        self.mesh = np.array(self.mesh) * 0.7

        self.cam_cx_1 = 322.52500
        self.cam_cy_1 = 244.11084
        self.cam_fx_1 = 591.01250
        self.cam_fy_1 = 590.16775

        self.cam_cx_2 = 319.5
        self.cam_cy_2 = 239.5
        self.cam_fx_2 = 577.5
        self.cam_fy_2 = 577.5

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.color = np.array([[255, 69, 0], [124, 252, 0], [0, 238, 238], [238, 238, 0], [155, 48, 255], [0, 0, 238], [255, 131, 250], [189, 183, 107], [165, 42, 42], [0, 234, 0]])

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.trancolor = transforms.ColorJitter(0.9, 0.5, 0.5, 0.05)
        self.length = count

        self.choose_obj = ''
        self.index = 0
        self.video_id = ''


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

    def get_pose(self, choose_frame, choose_obj):
        has_pose = []
        pose = {}
        with open('{0}/data/gts/real_test/results_real_test_{1}_{2}.pkl'.format(self.root, choose_frame.split("/")[-2], choose_frame.split("/")[-1]), 'rb') as f:
            nocs_data = cPickle.load(f)
        for idx in range(nocs_data['gt_RTs'].shape[0]):
            idx = idx + 1
            pose[idx] = nocs_data['gt_RTs'][idx-1]
            pose[idx][:3, :3] = pose[idx][:3, :3] / np.cbrt(np.linalg.det(pose[idx][:3, :3]))
            z_180_RT = np.zeros((4, 4), dtype=np.float32)
            z_180_RT[:3, :3] = np.diag([-1, -1, 1])
            z_180_RT[3, 3] = 1
            pose[idx] = z_180_RT @ pose[idx]
            pose[idx][:3,3] = pose[idx][:3,3] * 1000

        input_file = open('{0}_meta.txt'.format(choose_frame), 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            if input_line[-1] == choose_obj:
                ans = pose[int(input_line[0])]
                ans_idx = int(input_line[0])
                break
        input_file.close()

        ans = np.array(ans)
        ans_r = ans[:3, :3]
        ans_t = ans[:3, 3].flatten()

        return ans_r, ans_t, ans_idx


    def get_frame(self, choose_frame, choose_obj, syn_or_real, current_r, current_t):
        img = Image.open('{0}_color.png'.format(choose_frame))
        depth = np.array(self.load_depth('{0}_depth.png'.format(choose_frame)))

        if syn_or_real:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1
        else:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        cam_scale = 1.0

        target = []
        input_file = open('{0}/model_scales/{1}.txt'.format(self.root, choose_obj), 'r')
        for i in range(8):
            input_line = input_file.readline()
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        target = np.array(target)

        target = self.enlarge_bbox(copy.deepcopy(target))

        target_tmp = np.dot(target, current_r.T) + current_t
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

        cloud = np.dot(cloud - current_t, current_r)

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

        cloud = np.dot(cloud - current_t, current_r)

        cloud = cloud / 1000.0
        target = target / 1000.0

        return img, choose, cloud, target

    def re_scale(self, target_fr, target_to):
        ans_scale = target_fr / target_to
        ans_target = target_fr

        ans_scale = ans_scale[0][0]

        return ans_target, ans_scale


    def getone(self, current_r, current_t):
        choose_obj = self.choose_obj
        choose_frame = self.real_obj_list[self.choose_obj][self.index]

        split_name = choose_frame.split('/')
        for it in split_name:
            if it[:5] == 'scene':
                video_name = it

        print(video_name, self.choose_obj, self.index)

        if video_name != self.video_id:
            print(video_name, self.video_id)
            return 0

        img_fr, choose_fr, cloud_fr, target = self.get_frame(choose_frame, choose_obj, False, current_r, current_t)

        anchor_box, scale = self.get_anchor_box(target)
        cloud_fr = self.change_to_scale(scale, cloud_fr)


        return self.norm(torch.from_numpy(img_fr.astype(np.float32))).unsqueeze(0), \
               torch.LongTensor(choose_fr.astype(np.int32)).unsqueeze(0), \
               torch.from_numpy(cloud_fr.astype(np.float32)).unsqueeze(0), \
               torch.from_numpy(anchor_box.astype(np.float32)).unsqueeze(0), \
               torch.from_numpy(scale.astype(np.float32)).unsqueeze(0)


    def getfirst(self, choose_obj, video_id):
        self.choose_obj = choose_obj
        self.video_id = video_id
 

        for k in range(0, len(self.real_obj_list[self.choose_obj])):
            if self.video_id in self.real_obj_list[self.choose_obj][k].split('/'):
                self.index = k
                break

        current_r, current_t, _ = self.get_pose(self.real_obj_list[self.choose_obj][self.index], self.choose_obj)

        return current_r, current_t

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

    def projection(self, path, Kp, current_r, current_t, scale, www, add_on, score):
        img = np.array(Image.open('{0}_color.png'.format(self.real_obj_list[self.choose_obj][self.index])))

        cam_cx = self.cam_cx_2
        cam_cy = self.cam_cy_2
        cam_fx = self.cam_fx_2
        cam_fy = self.cam_fy_2
        cam_scale = 1.0

        target_r = current_r
        target_t = current_t

        target = []
        input_file = open('{0}/model_scales/{1}.txt'.format(self.root, self.choose_obj), 'r')
        for i in range(8):
            input_line = input_file.readline()
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        target = np.array(target)
        limit = search_fit(target)
        bbox = self.build_frame(limit[0], limit[1], limit[2], limit[3], limit[4], limit[5])
        anchor_box, scale = self.get_anchor_box(bbox)
        anchor_box = anchor_box * scale
        bbox = np.dot(bbox, target_r.T) + target_t
        bbox[:, 0] *= -1.0
        bbox[:, 1] *= -1.0

        anchor_box = np.dot(anchor_box, target_r.T) + target_t
        anchor_box[:, 0] *= -1.0
        anchor_box[:, 1] *= -1.0

        target = self.enlarge_bbox(copy.deepcopy(target))

        target = Kp.detach().cpu().numpy()[0] * 1000.0
        kkk = np.dot(target, target_r.T) + target_t

        if not add_on:
            fw = open('{0}_{1}_pose_False.txt'.format(path, self.index), 'w')
        else:
            fw = open('{0}_{1}_pose.txt'.format(path, self.index), 'w')

        for it in target_r:
            fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        it = target_t
        fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        fw.write('{0}\n'.format(score))
        fw.close()

        kkk[:, 0] *= -1.0
        kkk[:, 1] *= -1.0

        for tg in kkk:
            y = int(tg[0] * cam_fx / tg[2] + cam_cx)
            x = int(tg[1] * cam_fy / tg[2] + cam_cy)

            if x - 3 < 0 or x + 3 > 479 or y - 3 < 0 or y + 3 > 639:
                continue

            for xxx in range(x-3, x+4):
                for yyy in range(y-3, y+4):
                    img[xxx][yyy] = self.color[0]

        for tg in bbox:
            y = int(tg[0] * cam_fx / tg[2] + cam_cx)
            x = int(tg[1] * cam_fy / tg[2] + cam_cy)

            if x - 3 < 0 or x + 3 > 479 or y - 3 < 0 or y + 3 > 639:
                continue

            for xxx in range(x-2, x+3):
                for yyy in range(y-2, y+3):
                    img[xxx][yyy] = self.color[1]

        tg = anchor_box[www]

        y = int(tg[0] * cam_fx / tg[2] + cam_cx)
        x = int(tg[1] * cam_fy / tg[2] + cam_cy)

        if x - 5 >= 0 and x + 5 <= 479 and y - 5 >= 0 and y + 5 <= 639:
            for xxx in range(x-4, x+5):
                for yyy in range(y-4, y+5):
                    img[xxx][yyy] = self.color[2]

        if add_on:
            self.index += 1


    def __len__(self):
        return self.length

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
    #print(rmax - rmin, cmax - cmin)
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
