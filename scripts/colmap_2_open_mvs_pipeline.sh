#!/bin/bash

if [ "$#" != 1 ]; then
    echo "Usage: $0 <DATASET FOLDER>"
    exit 0
fi

## COLMAP
DATASETS_FOLDER="$1"
IMAGE_PATH="${DATASETS_FOLDER}/images"
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
    --database_path "${DATABASE_PATH}" \
    --image_path "${IMAGE_PATH}"

colmap exhaustive_matcher \
    --SiftMatching.use_gpu 0 \
    --database_path "${DATABASE_PATH}"

colmap mapper \
    --database_path "${DATABASE_PATH}" \
    --image_path "${IMAGE_PATH}" \
    --output_path "${SPARSE_PATH}"

colmap image_undistorter \
    --image_path "${IMAGE_PATH}" \
    --input_path "${SPARSE_PATH}/0" \
    --output_path "${DENSE_PATH}" \
    --output_type COLMAP \

colmap model_converter \
    --input_path "${SPARSE_PATH}/0" \
    --output_path "${SPARSE_PATH}"  \
    --output_type TXT

## OpenMVS
MVS_WS="${DATASETS_FOLDER}/mvs_ws"

mkdir -p "${MVS_WS}"
ln -s "${IMAGE_PATH}" "${MVS_WS}/images"

InterfaceCOLMAP \
    --working-folder "${MVS_WS}" \
    --input-file "${SPARSE_PATH}/0" \
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
