import os
import sys

import h5py
import torch
import tqdm

cfd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cfd)
wsd = os.path.join(cfd, '../../')
sys.path.append(wsd)


def names_to_pair(name0, name1):
    return '_'.join((name0, name1))


@torch.no_grad()
def spg(feature_path, pairs, matches_out, cfg):
    """Match features by SuperGlue"""

    from thirdparty.SuperGluePretrainedNetwork.models.superglue import SuperGlue as spg_matcher
    from utils import load_network

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running inference on device \"{device}\"')

    assert os.path.exists(feature_path), feature_path
    feature_file = h5py.File(feature_path, 'r')
    print(f'Exporting matches to {matches_out}')

    with open(pairs, 'r') as f:
        pair_list = f.read().rstrip('\n').split('\n')

    # load superglue model
    model = spg_matcher(cfg['conf']).to(device)
    model.eval()
    load_network(model, cfg['model']['path'], force=True)

    # match features by superglue
    match_file = h5py.File(matches_out, 'w')
    matched = set()
    for pair in tqdm.tqdm(pair_list):
        name0, name1 = pair.split(' ')
        pair = names_to_pair(name0, name1)

        if len({(name0, name1), (name1, name0)} & matched) \
            or pair in match_file:
            continue

        data = {}
        feats0, feats1 = feature_file[name0], feature_file[name1]
        for k in feats0.keys():
            data[k + '0'] = feats0[k].__array__()
        for k in feats1.keys():
            data[k + '1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(device) for k, v in data.items()}

        data['image0'] = torch.empty((1, 1, ) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1, ) + tuple(feats1['image_size'])[::-1])
        pred = model(data)

        grp = match_file.create_group(pair)
        matches0 = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches0)

        matches1 = pred['matches1'][0].cpu().short().numpy()
        grp.create_dataset('matches1', data=matches1)

        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)

        if 'matching_scores1' in pred:
            scores = pred['matching_scores1'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores1', data=scores)

        matched |= {(name0, name1), (name1, name0)}

    match_file.close()
    print('Finishing exporting matches...')


def main(features, pairs, matches_out, config):
    if os.path.isfile(matches_out):
        print('Old matches file exits, removing...')
        os.remove(matches_out)

    spg(features, pairs, matches_out, config)
