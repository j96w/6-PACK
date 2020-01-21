import os
import glob
import cv2
import numpy as np

intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

def backproject(depth, intrinsics, instance_mask):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    width = image_shape[1]
    height = image_shape[0]

    x = np.arange(width)
    y = np.arange(height)

    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    
    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz) #[num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]

    return pts, idxs

def load_depth(depth_path):
    depth = cv2.imread(depth_path, -1)

    if len(depth.shape) == 3:
        depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'

    return depth16

image_dir = "../My_NOCS/data/train"
scale_dir = "../My_NOCS/model_scales"
pose_dir = "../My_NOCS/data_pose/train"
folder_list = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
fnum = 0
num = 0
for folder in folder_list:
    fnum = fnum + 1
    image_list = glob.glob(os.path.join(image_dir, folder, '*_color.png'))
    image_list = sorted(image_list)
    for image_full_path in image_list:
        num = num + 1
        image_path = image_full_path.replace("_color.png", "")
        print(folder,fnum,num)
        mask_path = image_path + '_mask.png'
        if not os.path.exists(mask_path):
            continue
        depth_path = image_path + '_depth.png'
        meta_path = image_path + '_meta.txt'
        bbox_path = (image_path + '_bbox.txt').replace(image_dir,pose_dir)
        pose_path = (image_path + '_pose.txt').replace(image_dir,pose_dir)
        mask_im = cv2.imread(mask_path)[:, :, 2]
        mask_im = np.array(mask_im)
        inst_ids = np.unique(mask_im)
        pose_num = 0
        with open(meta_path, 'r') as f:
            with open(bbox_path, 'w') as bbox_f:
                for line in f:
                    line_info = line.split(' ')
                    inst_id = int(line_info[0])
                    cls_id = int(line_info[1])
                    model_id = str(line_info[-1]).replace("\n","")
                    if not inst_id in inst_ids:
                        continue
                    bbox_f.write(str(inst_id) + "\n")
                    if cls_id == 0:
                        for i in range(8):
                            bbox_f.write("0 0 0\n")
                        continue
                    depth_map = load_depth(depth_path)
                    tmp_mask = (mask_im == inst_id)
                    pts, idxs = backproject(depth_map, intrinsics, tmp_mask)
                    pose = []
                    with open(pose_path,'r') as posef:
                        for input_line in posef:
                            input_line = input_line.split(' ')
                            if len(input_line) == 1:
                                if int(input_line[0]) == inst_id:
                                    for i in range(4):
                                        input_line = posef.readline()
                                        if input_line[-1:] == '\n':
                                            input_line = input_line[:-1]
                                        input_line = input_line.split(' ')
                                        pose.append([float(input_line[0]), float(input_line[1]), float(input_line[2]), float(input_line[3])])
                                    break
                    pose = np.array(pose)
                    translation = pose[:3,3]
                    rotation = pose[:3,:3]
                    pts = pts - translation
                    pts = pts @ rotation
                    model_path = os.path.join(scale_dir, model_id) + ".txt"
                    bbox = np.loadtxt(model_path)
                    scale = (bbox[1] - bbox[0])/2
                    minx = np.min(pts[:,0])
                    miny = np.min(pts[:,1])
                    minz = np.min(pts[:,2])
                    maxx = np.max(pts[:,0])
                    maxy = np.max(pts[:,1])
                    maxz = np.max(pts[:,2])
                    x = max(maxx, abs(minx))
                    y = max(maxy, abs(miny))
                    z = max(maxz, abs(minz))
                    x_ratio = x / scale[2]
                    y_ratio = y / scale[1]
                    z_ratio = z / scale[0]
                    ratio = max(x_ratio, y_ratio, z_ratio)
                    scale = ratio * scale
                    bbox_f.write(str(scale[2]) + " " + str(scale[1]) + " " + str(scale[0]) + "\n")
                    bbox_f.write(str(scale[2]) + " " + str(scale[1]) + " " + str(-scale[0]) + "\n")
                    bbox_f.write(str(scale[2]) + " " + str(-scale[1]) + " " + str(scale[0]) + "\n")
                    bbox_f.write(str(scale[2]) + " " + str(-scale[1]) + " " + str(-scale[0]) + "\n")
                    bbox_f.write(str(-scale[2]) + " " + str(scale[1]) + " " + str(scale[0]) + "\n")
                    bbox_f.write(str(-scale[2]) + " " + str(scale[1]) + " " + str(-scale[0]) + "\n")
                    bbox_f.write(str(-scale[2]) + " " + str(-scale[1]) + " " + str(scale[0]) + "\n")
                    bbox_f.write(str(-scale[2]) + " " + str(-scale[1]) + " " + str(-scale[0]) + "\n")


                    

