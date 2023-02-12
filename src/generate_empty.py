import cv2
import logging
import os
import numpy as np

from pathlib import Path
from colmap.read_write_model import Camera, Image, Point3D
from colmap.read_write_model import rotmat2qvec
from colmap.read_write_model import write_model


def get_pose_from_txt(img_index, pose_dir):
    """ Read 4x4 transformation matrix from txt """
    pose_file = os.path.join(pose_dir, '{}.txt'.format(img_index))
    pose = np.loadtxt(pose_file)

    tvec = pose[:3, 3].reshape(3, )
    qvec = rotmat2qvec(pose[:3, :3]).reshape(4, )
    return pose, tvec, qvec


def get_intrin_from_txt(img_index, intrin_dir):
    """ Read 3x3 intrinsic matrix from txt """
    intrin_file = os.path.join(intrin_dir, '{}.txt'.format(img_index))
    intrin = np.loadtxt(intrin_file)

    return intrin


def import_data(img_lists, pose_dir, intrin_dir):
    """ Import intrinsics and camera pose info """
    points3D_out = {}
    images_out = {}
    cameras_out = {}

    def compare(img_name):
        key = img_name.split('/')[-1]
        return int(key.split('.')[0])
    img_lists.sort(key=compare)

    key, img_id, camera_id = 0, 0, 0
    xys_ = np.zeros((0, 2), float)
    point3D_ids_ = np.full(0, -1, int) # will be filled after triangulation

    # import data
    for img_path in img_lists:
        key += 1
        img_id += 1
        camera_id += 1

        img_name = Path(img_path).stem
        img_name_with_ext = Path(img_path).name

        # read pose
        _, tvec, qvec = get_pose_from_txt(img_name, pose_dir)

        # read intrinsic
        K = get_intrin_from_txt(img_name, intrin_dir)
        fx, fy, cx, cy = K[0][0], K[1][1], K[0, 2], K[1, 2]

        image = cv2.imread(img_path)
        h, w, _ = image.shape

        image = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=img_name_with_ext,
            xys=xys_,
            point3D_ids=point3D_ids_
        )

        camera = Camera(
            id=camera_id,
            model='PINHOLE',
            width=w,
            height=h,
            params=np.array([fx, fy, cx, cy])
        )

        images_out[key] = image
        cameras_out[key] = camera

    return cameras_out, images_out, points3D_out


def main(image_dir, empty_dir):
    img_lists = []
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1] in ['.jpg', '.png']:
            img_lists.append(os.path.join(image_dir, filename))
    img_lists = sorted(img_lists, key=lambda p: int(Path(p).stem))

    """ Write intrinsics and camera poses into COLMAP format model"""
    logging.info('Generate empty model...')
    sub_folder = image_dir.split('/')[-1]
    pose_dir = image_dir.replace(sub_folder, 'poses')
    intrin_dir = image_dir.replace(sub_folder, 'intrinsics')
    model = import_data(img_lists, pose_dir, intrin_dir)

    logging.info(f'Writing the COLMAP model to {empty_dir}')
    Path(empty_dir).mkdir(exist_ok=True, parents=True)

    write_model(*model, path=str(empty_dir), ext='.bin')
    logging.info('Finishing writing model.')
