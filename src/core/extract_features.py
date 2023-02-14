import os
from pathlib import Path
import sys

import h5py
import torch
import tqdm
from torch.utils.data import DataLoader

cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cfd)
wsd = os.path.join(cfd, '../../')
sys.path.append(wsd)


@torch.no_grad()
def spp(img_lists, feature_out, cfg):
    """extract keypoints info by superpoint"""

    from thirdparty.SuperPointPretrainedNetwork.superpoint import SuperPoint as spp_det
    from utils import NormalizedDataset, load_network

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running inference on device \"{device}\"')

    model = spp_det(cfg['conf']).to(device=device)
    model.eval()
    load_network(model, cfg['model']['path'], force=True)

    dataset = NormalizedDataset(img_lists, cfg['preprocessing'])
    loader = DataLoader(dataset, num_workers=1)

    feature_file = h5py.File(feature_out, 'w')
    print(f'Exporting features to {feature_out}')

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
    print('Finishing exporting features.')


def main(image_dir, feature_out, config):
    img_lists = []
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1] in ['.jpg', '.png']:
            img_lists.append(os.path.join(image_dir, filename))
    img_lists = sorted(img_lists, key=lambda p: int(Path(p).stem))

    if os.path.isfile(feature_out):
        print('Old feature file exits, removing...')
        os.remove(feature_out)

    spp(img_lists, feature_out, config)
