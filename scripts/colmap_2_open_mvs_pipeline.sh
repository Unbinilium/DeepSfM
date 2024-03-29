#!/bin/bash

if [ "$#" != 2 ]; then
    echo "Usage: $0 <ABSOULUTE DATASET FOLDER> <images/masked_images>"
    exit 0
fi

## COLMAP
DATASETS_FOLDER="$1"
IMAGE_PATH="${DATASETS_FOLDER}/$2"
SFM_WORKSPACE="${DATASETS_FOLDER}/sfm_ws"
DATABASE_PATH="${SFM_WORKSPACE}/database.db"
SPARSE_PATH="${SFM_WORKSPACE}/sparse"
DENSE_PATH="${SFM_WORKSPACE}/dense"

mkdir -p "${SFM_WORKSPACE}"
mkdir -p "${SPARSE_PATH}"
mkdir -p "${DENSE_PATH}"
cd "${SFM_WORKSPACE}"

colmap feature_extractor \
    --SiftExtraction.use_gpu 0 \
    --ImageReader.camera_model PINHOLE \
    --database_path "${DATABASE_PATH}" \
    --image_path "${IMAGE_PATH}" > "${SFM_WORKSPACE}/feature_extractor-$(date -I).log"

colmap exhaustive_matcher \
    --SiftMatching.use_gpu 0 \
    --database_path "${DATABASE_PATH}" > "${SFM_WORKSPACE}/exhaustive_matcher-$(date -I).log"

colmap mapper \
    --database_path "${DATABASE_PATH}" \
    --image_path "${IMAGE_PATH}" \
    --output_path "${SPARSE_PATH}" > "${SFM_WORKSPACE}/mapper-$(date -I).log"

LAST_SPARSE=""
for entry in "${SPARSE_PATH}"/*
do
    [ -d "$entry" ] && LAST_SPARSE="$entry"
done
echo "Last Sparse Folder: ${LAST_SPARSE}"

colmap image_undistorter \
    --image_path "${IMAGE_PATH}" \
    --input_path "${LAST_SPARSE}" \
    --output_path "${DENSE_PATH}" \
    --output_type COLMAP > "${SFM_WORKSPACE}/image_undistorter-$(date -I).log"

colmap model_converter \
    --input_path "${DENSE_PATH}/sparse" \
    --output_path "${SPARSE_PATH}" \
    --output_type TXT > "${SFM_WORKSPACE}/model_converter-$(date -I).log"

## OpenMVS
MVS_WS="${DATASETS_FOLDER}/mvs_ws"

mkdir -p "${MVS_WS}"
ln -s "${DENSE_PATH}/images" "${MVS_WS}/images"

InterfaceCOLMAP \
    --working-folder "${MVS_WS}" \
    --input-file "${DENSE_PATH}" \
    --output-file "${MVS_WS}/model_colmap.mvs"

DensifyPointCloud \
    --input-file "${MVS_WS}/model_colmap.mvs" \
    --working-folder "${MVS_WS}" \
    --output-file "${MVS_WS}/model_dense.mvs" \
    --archive-type -1

ReconstructMesh \
    --input-file "${MVS_WS}/model_dense.mvs" \
    --working-folder "${MVS_WS}" \
    --output-file "${MVS_WS}/model_dense_mesh.mvs"

RefineMesh \
    --resolution-level 1 \
    --input-file "${MVS_WS}/model_dense_mesh.mvs" \
    --working-folder "${MVS_WS}" \
    --output-file "${MVS_WS}/model_dense_mesh_refine.mvs"

TextureMesh \
    --export-type obj \
    --output-file "${MVS_WS}/model.obj" \
    --working-folder "${MVS_WS}" \
    --input-file "${MVS_WS}/model_dense_mesh_refine.mvs"
