import os
import numpy as np
import scipy.spatial.distance as distance

from pathlib import Path


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
    ext = Path(img_path).suffix
    pose_path = img_path.replace(ext, '.txt')
    pose_path = pose_path.replace('images', 'poses')
    return pose_path


def generate_pairs_from_poses(img_lists, pairs_out, num_matched, min_rotation=10):
    pose_lists = [get_pose_path(img_path) for img_path in img_lists]

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


def main(image_dir, pairs_out, config):
    img_lists = []

    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1] in ['.jpg', '.png']:
            img_lists.append(os.path.join(image_dir, filename))

    img_lists = sorted(img_lists, key=lambda p: int(Path(p).stem))

    if config['method'] == 'sequential':
        generate_sequential_pairs(img_lists, pairs_out)
    elif config['method'] == 'from-poses':
        generate_pairs_from_poses(img_lists, pairs_out, config['num_matched'], config['min_rotation'])
    else:
        raise NotImplementedError()
