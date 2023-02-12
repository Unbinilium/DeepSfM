import h5py
import logging
import os
import tqdm
import torch

from pathlib import Path
from torch.utils.data import DataLoader


@torch.no_grad()
def spp(img_lists, feature_out, cfg):
    """extract keypoints info by superpoint"""
    from utils import load_network
    from utils import NormalizedDataset
    from thirdparty.SuperPointPretrainedNetwork.superpoint import SuperPoint as spp_det

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Running inference on device \"{}\"'.format(device))

    model = spp_det(cfg['conf']).to(device=device)
    model.eval()
    load_network(model, cfg['model']['path'], force=True)

    dataset = NormalizedDataset(img_lists, cfg['preprocessing'])
    loader = DataLoader(dataset, num_workers=1)

    feature_file = h5py.File(feature_out, 'w')
    logging.info(f'Exporting features to {feature_out}')
    for data in tqdm.tqdm(loader):
        inp = data['image'].to(device=device)
        pred = model(inp)

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        pred['image_size'] = data['size'][0].numpy()

        grp = feature_file.create_group(Path(data['path'][0]).name)
        for k, v in pred.items():
            grp.create_dataset(k, data=v)

        del pred

    feature_file.close()
    logging.info('Finishing exporting features.')


def main(image_dir, feature_out, config):
    img_lists = []

    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1] in ['.jpg', '.png']:
            img_lists.append(os.path.join(image_dir, filename))

    img_lists = sorted(img_lists, key=lambda p: int(Path(p).stem))
    spp(img_lists, feature_out, config)
