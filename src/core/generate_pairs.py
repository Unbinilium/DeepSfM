import os
from pathlib import Path

import numpy as np
import torch
import scipy.spatial.distance as distance


def get_pairswise_distances(pose_files):
    Rs = []
    ts = []

    seqs_ids = {}
    for i in range(len(pose_files)):
        pose_file = pose_files[i]
        seq_name = pose_file.split('/')[-3]
        if seq_name not in seqs_ids.keys():
            seqs_ids[seq_name] = [i]
        else:
            seqs_ids[seq_name].append(i)

    for pose_file in pose_files:
        pose = np.loadtxt(pose_file)
        R = pose[:3, :3]
        t = pose[:3, 3:]
        Rs.append(R)
        ts.append(t)

    Rs = np.stack(Rs, axis=0)
    ts = np.stack(ts, axis=0)

    Rs = Rs.transpose(0, 2, 1) # [n, 3, 3]
    ts = -(Rs @ ts)[:, :, 0] # [n, 3, 3] @ [n, 3, 1]

    dist = distance.squareform(distance.pdist(ts))
    trace = np.einsum('nji,mji->mn', Rs, Rs, optimize=True)
    dR = np.clip((trace - 1) / 2, -1., 1.)
    dR = np.rad2deg(np.abs(np.arccos(dR)))

    return dist, dR, seqs_ids


def generate_sequential_pairs(img_lists, pairs_out):
    pairs = []

    for i in range(1, len(img_lists) + 1):
        name0 = Path(img_lists[i - 1]).name
        name1 = Path(img_lists[i % len(img_lists)]).name

        pairs.append((name0, name1))

    with open(pairs_out, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


def get_pose_path(img_path):
    img_name = Path(img_path).stem
    datasets_dir = Path(img_path).parent.parent
    pose_name_ext = f'{img_name}.txt'
    pose_path = os.path.join(datasets_dir, f'poses/{pose_name_ext}')

    return pose_path


def generate_pairs_from_poses(img_lists, pairs_out, num_matched, min_rotation=10):
    pose_lists = [get_pose_path(img_path) for img_path in img_lists]
    print(f'Got {len(pose_lists)} poses...')

    print(f'Getting pairwise distances...')
    dist, dR, seqs_ids = get_pairswise_distances(pose_lists)

    valid = dR > min_rotation
    np.fill_diagonal(valid, False)
    dist = np.where(valid, dist, np.inf)

    pairs = []
    num_matched_per_seq = num_matched // len(seqs_ids.keys())
    for i in range(len(img_lists)):
        dist_i = dist[i]
        for seq_id in seqs_ids:
            ids = np.array(seqs_ids[seq_id])
            try:
                idx = np.argpartition(dist_i[ids], num_matched_per_seq * 2)[: num_matched_per_seq:2]
            except:
                idx = np.argpartition(dist_i[ids], dist_i.shape[0]-1)
            idx = ids[idx]
            idx = idx[np.argsort(dist_i[idx])]
            idx = idx[valid[i][idx]]

            for j in idx:
                name0 = Path(img_lists[i]).name
                name1 = Path(img_lists[j]).name

                pairs.append((name0, name1))

    with open(pairs_out, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
        print('Finishing exporting pairs...')


@torch.no_grad()
def generate_pairs_exhaustive(img_lists, pairs_out, num_matched, feature_path, cfg):
    import itertools as it

    import h5py
    from pathlib import Path
    import tqdm
    from thirdparty.SuperGluePretrainedNetwork.models.superglue import SuperGlue as spg_matcher
    from utils import load_network

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running inference on device \"{device}\"')

    assert os.path.exists(feature_path), feature_path
    feature_file = h5py.File(feature_path, 'r')
    print(f'Loaded features from {feature_path}')

    model = spg_matcher(cfg['conf']).to(device)
    model.eval()
    load_network(model, cfg['model']['path'], force=True)

    idx_pairs = [e for e in it.permutations(np.arange(len(img_lists)), 2)]
    print(f'Calculated {len(idx_pairs)} index pairs...')

    print('Exhaustive matching...')
    matches_of_pairs = {}
    for idx_pair in tqdm.tqdm(idx_pairs):
        i, j = idx_pair
        name0, name1 = Path(img_lists[i]).name, Path(img_lists[j]).name
        feats0, feats1 = feature_file[name0], feature_file[name1]

        data = {}
        for k in feats0.keys():
            data[k + '0'] = feats0[k].__array__()
        for k in feats1.keys():
            data[k + '1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(device) for k, v in data.items()}

        data['image0'] = torch.empty((1, 1, ) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1, ) + tuple(feats1['image_size'])[::-1])
        pred = model(data)

        pred_matches = pred['matches0'][0].detach().cpu().numpy()
        valid = pred_matches > -1
        matches = pred['matching_scores0'][0].detach().cpu().numpy()[valid]

        pair = '_'.join([str(i), str(j)])
        matches_of_pairs[pair] = matches

    print('Filtering pairs...')
    good_pairs = []
    for i in range(0, len(img_lists)):
        p_v_s = {}
        len_sum = 0
        for j in range(0, len(img_lists)):
            if i == j:
                continue
            pair = '_'.join([str(i), str(j)])
            m = matches_of_pairs[pair]
            p_v_s[pair] = len(m) * np.average(m)
            len_sum += len(m)
        if len_sum == 0:
            continue
        for (k, v) in p_v_s.items():
            p_v_s[k] = v / np.float32(len_sum)
        candidate = []
        for (k, _) in sorted(p_v_s.items(), key=lambda item: item[1], reverse=True):
            p = k.split('_')[:2]
            candidate.append((int(p[0]), int(p[1])))
        good_pairs.extend(candidate[:num_matched])

    with open(pairs_out, 'w') as f:
        f.write('\n'.join('{} {}'.format(Path(img_lists[i]).name, Path(img_lists[j]).name) for (i, j) in good_pairs))
        print('Finishing exporting pairs...')


def main(image_dir, pairs_out, config):
    img_lists = []
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1] in ['.jpg', '.png']:
            img_lists.append(os.path.join(image_dir, filename))
    img_lists = sorted(img_lists, key=lambda p: int(Path(p).stem))

    if os.path.isfile(pairs_out):
        print('Old pairs file exits, removing...')
        os.remove(pairs_out)

    if config['method'] == 'sequential':
        generate_sequential_pairs(img_lists, pairs_out)
    elif config['method'] == 'from-poses':
        generate_pairs_from_poses(img_lists, pairs_out, config['num_matched'], config['min_rotation'])
    elif config['method'] == 'exhaustive':
        generate_pairs_exhaustive(img_lists, pairs_out, config['num_matched'], config['feature_path'], config['superglue'])
    else:
        raise NotImplementedError()
