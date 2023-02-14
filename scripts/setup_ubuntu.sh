#!/bin/bash


## Get workspace path
WS="$(pwd)"


## Update sources
sudo apt-get update


## Install python deps
pip3 install -r "${WS}/env/venv_cpu_3.10.txt"


## Compile COLMAP
sudo apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-iostreams-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libopencv-dev

[ -d /tmp/colmap ] && rm -fr /tmp/colmap
cp -r "${WS}/thirdparty/colmap" /tmp/colmap
cd /tmp/colmap

git checkout dev
mkdir build
cd build
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES='native'
ninja
sudo ninja install


## Compile OpenMVS
sudo apt-get install -y \
    libboost-iostreams-dev \
    libglfw3-dev libopencv-dev

[ -d /tmp/VCG ] && rm -fr /tmp/VCG
cp -r "${WS}/thirdparty/VCG" /tmp/VCG

[ -d /tmp/openMVS ] && rm -fr /tmp/openMVS
cp -r "${WS}/thirdparty/openMVS" /tmp/openMVS
cd /tmp/openMVS

mkdir make
cd make

cmake .. -DVCG_ROOT="/tmp/VCG" -DOpenMVS_USE_CUDA=OFF
cmake --build . -j4
sudo cmake --install .

echo "PATH=/usr/local/bin/OpenMVS:\$PATH" >> ~/.profile
source ~/.profile


## Install meshlab
sudo apt-get install -y \
    meshlab
