import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision import transforms

cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cfd)
wsd = os.path.join(cfd, '../../')
sys.path.append(wsd)

from thirdparty.DIS.IS_Net.data_loader_cache import im_preprocess, normalize


def build_model(hypar, device):
    net = hypar['model'] # GOSNETINC(3, 1)
    # convert to half precision
    if(hypar['model_digit'] == 'half'):
        net.half()
        for layer in net.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.float()
    net.to(device)
    if(hypar['restore_model'] != ''):
        net.load_state_dict(torch.load(hypar['model_path'] + '/' + hypar['restore_model'], map_location=device))
        net.to(device)
    net.eval()
    return net


class GOSNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


def normalize_2_tensor_with_size(im, transform, cache_size):
    im, im_shp = im_preprocess(im, cache_size)
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)


def dis_predict_mask(dis_net, model_digit, device, image_tensor_with_size):
    inputs_val, shapes_val = image_tensor_with_size

    inputs_val = inputs_val.type(torch.FloatTensor if model_digit == 'full' else torch.HalfTensor)
    inputs_val_v = torch.autograd.Variable(inputs_val, requires_grad=False).to(device) # wrap inputs in Variable

    ds_val = dis_net(inputs_val_v)[0] # list of 6 results

    pred_val = ds_val[0][0, :, :, :] # B x 1 x H x W, we want the first one which is the most accurate prediction
    # recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(torch.nn.functional.upsample(
        torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]),
        mode='bilinear'))
    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi) # max = 1
    mask = (pred_val.detach().cpu().numpy() * 255).astype(np.uint8) # it is the mask we need

    return mask


@torch.no_grad()
def generate_masks_and_masked_images(img_lists, masks_out, masked_images_out, hypar):
    print('Initializing DIS Network...')

    from thirdparty.DIS.IS_Net.models.isnet import ISNetDIS
    hypar['model'] = ISNetDIS()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running inference on device \"{device}\"')

    dis_net_full = build_model(hypar, device)
    dis_net_full.eval()

    transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])

    print('Generating masks and masked images...')
    for img_path in tqdm.tqdm(img_lists):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        inp_tensor_with_size = normalize_2_tensor_with_size(img, transform, hypar['cache_size'])
        mask = dis_predict_mask(dis_net_full, hypar['model_digit'], device, inp_tensor_with_size)

        mask_name = Path(img_path).stem
        cv2.imwrite(os.path.join(masks_out, mask_name + '.png'), mask)

        rgba_img = Image.fromarray(img).convert('RGBA')
        rgba_img.putalpha(Image.fromarray(mask).convert('L'))
        rgba_img.save(os.path.join(masked_images_out, mask_name + '.png'))


def main(image_dir, masks_out, masked_images_out, config):
    if not os.path.isdir(masks_out):
        os.makedirs(masks_out)
    else:
        print(f'Old masks out directory exist: {masks_out}')

    if not os.path.isdir(masked_images_out):
        os.makedirs(masked_images_out)
    else:
        print(f'Old masked images out directory exist: {masked_images_out}')

    img_lists = []

    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1] in ['.jpg', '.png']:
            img_lists.append(os.path.join(image_dir, filename))

    img_lists = sorted(img_lists, key=lambda p: int(Path(p).stem))

    generate_masks_and_masked_images(img_lists, masks_out, masked_images_out, config)
