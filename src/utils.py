import cv2
import numpy as np
import os
import torch

from types import SimpleNamespace
from torch.utils.data import Dataset


class NormalizedDataset(Dataset):
    """read images(suppose images have been cropped)"""
    default_conf = {
        'globs': ['*.jpg', '*.png'],
        'grayscale': True,
    }

    def __init__(self, img_lists, conf):
        self.img_lists = img_lists
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})

        if len(img_lists) == 0:
            raise ValueError('Could not find any image.')

    def __getitem__(self, index):
        img_path = self.img_lists[index]

        mode = cv2.IMREAD_GRAYSCALE if self.conf.grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(img_path, mode)
        size = image.shape[:2]

        image = image.astype(np.float32)
        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))
        image /= 255.

        data = {
            'path': str(img_path),
            'image': image,
            'size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.img_lists)


def load_network(net, model_dir, resume=True, epoch=-1, strict=True, force=False):
    """Load latest network-weights from dir or path"""
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        if force:
            raise NotImplementedError
        else:
            print('pretrained model does not exist')
            return 0

    if os.path.isdir(model_dir):
        pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if 'pth' in pth]
        if len(pths) == 0:
            return 0
        if epoch == -1:
            pth = max(pths)
        else:
            pth = epoch
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('=> load weights: ', model_path)
    pretrained_model = torch.load(model_path, torch.device('cpu'))
    if 'net' in pretrained_model.keys():
        net.load_state_dict(pretrained_model['net'], strict=strict)
    else:
        net.load_state_dict(pretrained_model, strict=strict)
    return pretrained_model.get('epoch', 0) + 1
