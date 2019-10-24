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
import copy
from libs.tracker import tracker
import math

#start and end frame
st = 
ed = 

# category id
category_id = 
tracker = tracker(category_id)

# provide the initial pose and scale of the object
current_r, current_t, bbox = 

img_dir = 'rgb/{0}.png'.format(st)
depth_dir = 'depth/{0}.npy'.format(st)
current_r, current_t = tracker.init_estimation(img_dir, depth_dir, current_r, current_t, bbox)

for i in range(st+1, ed):
    img_dir = 'rgb/{0}.png'.format(i)
    depth_dir = 'depth/{0}.npy'.format(i)
    current_r, current_t = tracker.next_estimation(img_dir, depth_dir, current_r, current_t)