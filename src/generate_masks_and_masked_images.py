import cv2
import numpy as np
import os
import tqdm
import torch

from pathlib import Path
from torchvision import transforms
from thirdparty.DIS.IS_Net.data_loader_cache import normalize, im_reader, im_preprocess
from thirdparty.DIS.IS_Net.models.isnet import ISNetGTEncoder, ISNetDIS


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running inference on device \'{}\''.format(device))


hypar = {
    'model_path': '../data/models/is-net',  # load trained weights from this path
    'restore_model': 'isnet.pth',           # name of the to-be-loaded weights
    'interm_sup': False,                    # indicate if activate intermediate feature supervision
    'model_digit': 'full',                  # indicates 'half' or 'full' accuracy of float number
    'seed': 0,
    'cache_size': [1024, 1024],             # cached input spatial resolution, can be configured into different size
    'input_size': [1024, 1024],             # mdoel input spatial size, usually use the same value hypar['cache_size'], which means we don't further resize the images
    'crop_size': [1024, 1024],              # random crop size from the input, it is usually set as smaller than hypar['cache_size'], e.g., [920,920] for data augmentation
}


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


transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])


def normalize_2_tensor_with_size(im, hypar):
    im, im_shp = im_preprocess(im, hypar['cache_size'])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)


def dis_predict_mask(dis_net, image_tensor_with_size):
    inputs_val, shapes_val = image_tensor_with_size

    inputs_val = inputs_val.type(torch.FloatTensor if hypar['model_digit'] == 'full' else torch.HalfTensor)
    inputs_val_v = torch.autograd.Variable(inputs_val, requires_grad=False).to(device) # wrap inputs in Variable

    ds_val = dis_net(inputs_val_v)[0] # list of 6 results

    pred_val = ds_val[0][0, :, :, :] # B x 1 x H x W, we want the first one which is the most accurate prediction
    # recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(torch.nn.functional.upsample(
        torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]),
        mode='bilinear')
    )
    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi) # max = 1
    mask = (pred_val.detach().cpu().numpy() * 255).astype(np.uint8) # it is the mask we need

    return mask


@torch.no_grad()
def gm_and_mi(img_lists, masks_out, masked_images_out):
    hypar['model'] = ISNetDIS()
    dis_net_full = build_model(hypar, device)
    dis_net_full.eval()

    for img_path in tqdm.tqdm(img_lists):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        inp_tensor_with_size = normalize_2_tensor_with_size(img, hypar)
        mask = dis_predict_mask(dis_net_full, inp_tensor_with_size)

        mask_name = os.path.basename(img_path)[0]
        cv2.imwrite(os.path.join(masks_out, mask_name + '.png'), mask)

        masked_img = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(os.path.join(masked_images_out, mask_name + '.png'), masked_img)


def main(image_dir, masks_out, masked_images_out):
    if not os.path.isdir(masks_out):
        os.makedirs(masks_out)

    if not os.path.isdir(masked_images_out):
        os.makedirs(masked_images_out)

    img_lists = []

    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1] in ['.jpg', '.png']:
            img_lists.append(os.path.join(image_dir, filename))

    img_lists = sorted(img_lists, key=lambda p: int(Path(p).stem))
    gm_and_mi(img_lists, masks_out, masked_images_out)
