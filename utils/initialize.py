import torch
import torch.nn as nn
import numpy as np
import os
import random
import config
import shutil


def random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def create_image_folders():
    for i in range(config.Hyperparameter.NUM_CLASS):
        dirpath = config.Constant.IMAGE_FOLDER + str(i)
        is_exist = os.path.exists(dirpath)
        if not is_exist:
            os.makedirs(dirpath)


def delete_images():
    for dirpath, dirnames, filenames in os.walk(config.Constant.IMAGE_FOLDER):
        for d in dirnames:
            shutil.rmtree(os.path.join(dirpath, d))
