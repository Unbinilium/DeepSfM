import os

from pathlib import Path


def generate_pairs(img_lists, covis_pairs_out, cfg):
    if cfg['method'] == 'sequential':
        pairs = []

        for i in range(1, len(img_lists) + 1):
            name0 = Path(img_lists[i - 1]).stem
            name1 = Path(img_lists[i % len(img_lists)]).stem

            pairs.append((name0, name1))

        with open(covis_pairs_out, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

    else:
        raise NotImplementedError


def main(image_dir, covis_pairs_out, config):
    img_lists = []

    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1] in ['.jpg', '.png']:
            img_lists.append(os.path.join(image_dir, filename))

    img_lists = sorted(img_lists, key=lambda p: int(Path(p).stem))

    generate_pairs(img_lists, covis_pairs_out, config)
